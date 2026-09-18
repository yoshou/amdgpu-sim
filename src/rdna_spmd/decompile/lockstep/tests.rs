use super::*;
use crate::rdna_spmd::compiler::{compile_lockstep, compile_scalar};
use crate::rdna_spmd::engine::kernel::Code;
use crate::rdna_spmd::ir::{parse, Func, Inst, IntOp, IntPred, Op, PacketOp, Term, Ty, ValueId};
use crate::rdna_spmd::program::{Parameter, ParameterSource, Program};
use std::sync::Arc;

/// Every lane program here reads the work item from VGPR 0 -- the lane's
/// index, which it scales to the offset of the lane's slot -- the output
/// buffer's address from SGPRs 0 and 1, a value every lane shares from SGPR 2,
/// and EXEC.
fn lane(text: &str) -> LiftedFunction {
    let registry = crate::rdna_spmd::targets::rdna4::registry();
    let ir = parse::func(&registry, text).unwrap();
    let input = |source, ty| Parameter { source, ty };
    LiftedFunction {
        registry: Arc::new(registry),
        ir,
        parameter_inputs: vec![
            input(ParameterSource::Vgpr(0), Ty::I32),
            input(ParameterSource::Sgpr(0), Ty::I32),
            input(ParameterSource::Sgpr(1), Ty::I32),
            input(ParameterSource::Sgpr(2), Ty::I32),
            input(ParameterSource::MaskBit(126), Ty::I1),
        ],
        revision: 0,
    }
}

const UNTOUCHED: u32 = 0xdead_beef;

/// The words in the record a program reads from the start of the buffer.
const RECORD: usize = 8;

/// Runs the lane program alone and in packets of every width, with `shared`
/// in SGPR 2, and checks what each lane stored at its own slot of the output.
fn check(lane: &LiftedFunction, shared: u32, expected: impl Fn(u32) -> Option<u32>) {
    // At least a record's worth of slots, so a program can read words from
    // the buffer that no lane of a narrow packet writes.
    check_over(lane, shared, RECORD, |_| UNTOUCHED, expected)
}

/// As `check`, over a buffer of at least `words` words holding `initial`.
fn check_over(
    lane: &LiftedFunction,
    shared: u32,
    words: usize,
    initial: impl Fn(usize) -> u32,
    expected: impl Fn(u32) -> Option<u32>,
) {
    for width in [0u32, 1, 2, 4, 8, 16, 32] {
        let lanes = width.max(1) as usize;
        let mut output: Vec<u32> = (0..lanes.max(words)).map(&initial).collect();
        let mut sgprs = [0u32; 128];
        let address = output.as_mut_ptr() as u64;
        sgprs[0] = address as u32;
        sgprs[1] = (address >> 32) as u32;
        sgprs[2] = shared;
        let mut vgprs = vec![0u32; 256 * lanes];
        // The dispatch gives each lane its work item: x is the lane's index
        // in a row that holds the packet.
        for l in 0..lanes {
            vgprs[l] = l as u32;
        }
        unsafe {
            if width == 0 {
                compile_scalar(
                    Program {
                        function: lane.clone(),
                    },
                    256,
                )
                .run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0);
            } else {
                let alone = super::super::Lane::from(lane.clone());
                let (Code::Packet(kernel), _) = compile_lockstep(&alone, 256, width, None)
                    .expect("every lane is at each operation over the wave")
                else {
                    panic!("a lane program that keeps no wave operation runs alone")
                };
                kernel.run(sgprs.as_mut_ptr(), vgprs.as_mut_ptr(), 0, 0, u32::MAX, 0);
            }
        }
        let wanted: Vec<u32> = (0..lanes as u32)
            .map(|l| expected(l).unwrap_or(initial(l as usize)))
            .collect();
        assert_eq!(output[..lanes], wanted, "width {}", width);
    }
}

fn lowered(lane: &LiftedFunction) -> Func {
    lockstep(
        &super::super::Lane::from(lane.clone()),
        Packing {
            lanes: 16,
            aligned: true,
        },
    )
    .expect("every lane is at each operation over the wave")
    .ir
}

fn definitions(f: &Func) -> Vec<Option<Op>> {
    f.definitions()
}

/// The value a block parameter stands for when the block has one way in.
fn source(f: &Func, v: ValueId) -> ValueId {
    for (&id, block) in &f.blocks {
        let Some(index) = block.params.iter().position(|p| p.0 == v) else {
            continue;
        };
        let incoming: Vec<ValueId> = f
            .blocks
            .values()
            .flat_map(|b| b.term.edges())
            .filter(|e| e.dst == id)
            .map(|e| e.args[index])
            .collect();
        return match incoming.as_slice() {
            [only] => source(f, *only),
            _ => v,
        };
    }
    v
}

fn insts(f: &Func) -> impl Iterator<Item = &Inst> {
    f.blocks.values().flat_map(|b| &b.insts)
}

/// Whether a bit is `condition` held to the lanes with work items: the
/// condition itself, or the condition and the EXEC the packet started with.
fn holds_under_valid(f: &Func, defs: &[Option<Op>], bit: ValueId, condition: ValueId) -> bool {
    let valid = f.blocks[&f.entry].params[4].0;
    let same = |v: ValueId| source(f, v) == condition;
    same(bit)
        || matches!(defs[source(f, bit).0], Some(Op::Int(IntOp::And, a, b))
            if (source(f, a) == valid && same(b)) || (source(f, b) == valid && same(a)))
}

fn queries(f: &Func) -> usize {
    insts(f)
        .filter(|inst| {
            matches!(
                inst,
                Inst::Packet {
                    op: PacketOp::Any,
                    ..
                }
            )
        })
        .count()
}

fn selects(f: &Func) -> Vec<(ValueId, ValueId, ValueId)> {
    insts(f)
        .filter_map(|inst| match inst {
            Inst::Core {
                op: Op::Select(c, a, b),
                ..
            } => Some((*c, *a, *b)),
            _ => None,
        })
        .collect()
}

#[test]
fn lanes_leaving_a_loop_after_different_trips_keep_what_they_computed() {
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v40: i32 = const i32 0x2
           v41: i32 = int shl v0, v40
           v5: i32 = const i32 0x2
           v6: i32 = int lshr v41, v5
           v7: i32 = const i32 0x0
           br b1(v6, v7, v6, v41, v1, v2, v4)
         b1(v8: i32, v9: i32, v10: i32, v11: i32, v12: i32, v13: i32, v14: i1):
           v15: i32 = const i32 0x0
           v16: i1 = cmp eq v8, v15
           condbr v16, b3(v9, v10, v11, v12, v13, v14), b2(v8, v9, v10, v11, v12, v13, v14)
         b2(v17: i32, v18: i32, v19: i32, v20: i32, v21: i32, v22: i32, v23: i1):
           v24: i32 = const i32 0x1
           v25: i32 = int sub v17, v24
           v26: i32 = const i32 0x3
           v27: i32 = int add v18, v26
           br b1(v25, v27, v19, v20, v21, v22, v23)
         b3(v28: i32, v29: i32, v30: i32, v31: i32, v32: i32, v33: i1):
           v34: i64 = pack64 v31, v32
           v35: i64 = convert zext i64 v30
           v36: i64 = int add v34, v35
           v37: i32 = const i32 0x7
           v38: i32 = int mul v29, v37
           v39: i32 = int add v38, v28
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v36, v39, v33)
           ret",
    );
    check(&lane, 0, |l| Some(10 * l));
}

#[test]
fn a_lane_leaving_with_what_it_would_go_around_with_keeps_it_in_the_loop_parameter() {
    // The latch sends i + 1 and acc + 3 both around and out, so the loop keeps
    // a leaving lane's values in its own parameters, as masked stores to a
    // variable would, and needs no slots of their own for them.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v32: i32 = const i32 0x2
           v33: i32 = int shl v0, v32
           v5: i32 = const i32 0x2
           v6: i32 = int lshr v33, v5
           v7: i32 = const i32 0x0
           br b1(v7, v7, v6, v33, v1, v2, v4)
         b1(v8: i32, v9: i32, v10: i32, v11: i32, v12: i32, v13: i32, v14: i1):
           v15: i32 = const i32 0x1
           v16: i32 = int add v8, v15
           v17: i32 = const i32 0x3
           v18: i32 = int add v9, v17
           v19: i1 = cmp ugt v16, v10
           condbr v19, b2(v18, v16, v11, v12, v13, v14), b1(v16, v18, v10, v11, v12, v13, v14)
         b2(v20: i32, v21: i32, v22: i32, v23: i32, v24: i32, v25: i1):
           v26: i64 = pack64 v23, v24
           v27: i64 = convert zext i64 v22
           v28: i64 = int add v26, v27
           v29: i32 = const i32 0x10
           v30: i32 = int mul v20, v29
           v31: i32 = int add v30, v21
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v28, v31, v25)
           ret",
    );
    let packet = lowered(&lane);
    let (&id, header) = packet
        .blocks
        .iter()
        .find(|(&id, b)| b.term.edges().any(|e| e.dst == id))
        .expect("the packet program keeps the loop");
    let back = header.term.edges().find(|e| e.dst == id).unwrap();
    let changing = header
        .params
        .iter()
        .zip(&back.args)
        .filter(|(p, arg)| p.1 == Ty::I32 && p.0 != **arg)
        .count();
    assert_eq!(
        changing, 2,
        "only i and acc change as the loop goes around; nothing else holds what a lane left with"
    );
    check(&lane, 0, |l| Some(49 * (l + 1)));
}

#[test]
fn a_loop_every_lane_leaves_together_is_an_ordinary_loop() {
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v33: i32 = const i32 0x2
           v34: i32 = int shl v0, v33
           v5: i32 = const i32 0x0
           br b1(v5, v5, v34, v3, v1, v2, v4)
         b1(v6: i32, v7: i32, v8: i32, v9: i32, v10: i32, v11: i32, v12: i1):
           v13: i1 = cmp ult v6, v9
           condbr v13, b2(v6, v7, v8, v9, v10, v11, v12), b3(v7, v8, v10, v11, v12)
         b2(v14: i32, v15: i32, v16: i32, v17: i32, v18: i32, v19: i32, v20: i1):
           v21: i32 = int mul v14, v16
           v22: i32 = int add v15, v21
           v23: i32 = const i32 0x1
           v24: i32 = int add v14, v23
           br b1(v24, v22, v16, v17, v18, v19, v20)
         b3(v25: i32, v26: i32, v27: i32, v28: i32, v29: i1):
           v30: i64 = pack64 v27, v28
           v31: i64 = convert zext i64 v26
           v32: i64 = int add v30, v31
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v32, v25, v29)
           ret",
    );
    let packet = lowered(&lane);
    assert_eq!(queries(&packet), 0, "the trip count is a scalar test");
    assert!(
        selects(&packet).is_empty(),
        "no lane leaves early, so nothing is kept for it"
    );
    check(&lane, 5, |l| Some(40 * l));
}

#[test]
fn a_merge_after_a_branch_the_lanes_disagree_on_chooses_on_its_condition() {
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v34: i32 = const i32 0x2
           v35: i32 = int shl v0, v34
           v5: i32 = const i32 0x0
           v6: i1 = cmp ne v35, v5
           condbr v6, b1(v35, v1, v2, v4), b5(v4)
         b1(v7: i32, v8: i32, v9: i32, v10: i1):
           v11: i32 = const i32 0xc
           v12: i1 = cmp ult v7, v11
           condbr v12, b2(v7, v8, v9, v10), b3(v7, v8, v9, v10)
         b2(v13: i32, v14: i32, v15: i32, v16: i1):
           v17: i32 = const i32 0x64
           v18: i32 = int add v13, v17
           br b4(v18, v13, v14, v15, v16)
         b3(v19: i32, v20: i32, v21: i32, v22: i1):
           v23: i32 = const i32 0x3
           v24: i32 = int mul v19, v23
           br b4(v24, v19, v20, v21, v22)
         b4(v25: i32, v26: i32, v27: i32, v28: i32, v29: i1):
           v30: i64 = pack64 v27, v28
           v31: i64 = convert zext i64 v26
           v32: i64 = int add v30, v31
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v32, v25, v29)
           ret
         b5(v33: i1):
           ret",
    );
    let packet = lowered(&lane);
    let defs = definitions(&packet);
    let chosen: Vec<_> = selects(&packet)
        .into_iter()
        .filter(|&(_, a, _)| packet.types[a.0] == Ty::I32)
        .collect();
    assert_eq!(chosen.len(), 1);
    assert!(
        matches!(
            defs[source(&packet, chosen[0].0).0],
            Some(Op::Cmp(IntPred::Ult, ..))
        ),
        "the lanes that skipped the branch play no part in the choice"
    );
    check(&lane, 0, |l| match l {
        0 => None,
        1 | 2 => Some(4 * l + 100),
        _ => Some(12 * l),
    });
}

/// How many multiply-and-add steps the costly arm takes: more than a query
/// over the lanes costs.
const STEPS: u32 = 60;

/// A chain of multiply-and-add steps from `from`, numbering its values from
/// `next`; the last step's value is `next + 4 * STEPS - 1`.
fn chain(from: usize, next: usize) -> String {
    let mut body = String::new();
    let mut last = from;
    for step in 0..STEPS as usize {
        let at = next + 4 * step;
        body.push_str(&format!(
            "v{k}: i32 = const i32 0x3
             v{m}: i32 = int mul v{last}, v{k}
             v{one}: i32 = const i32 0x1
             v{sum}: i32 = int add v{m}, v{one}
             ",
            k = at,
            m = at + 1,
            one = at + 2,
            sum = at + 3,
            last = last
        ));
        last = at + 3;
    }
    body
}

/// An arm that computes a long chain before it stores.
fn costly_arm() -> String {
    let body = chain(6, 14);
    let last = 14 + 4 * STEPS - 1;
    let (two, slot) = (last + 1, last + 2);
    format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v{two}: i32 = const i32 0x2
           v{slot}: i32 = int shl v0, v{two}
           v5: i1 = cmp ugt v{slot}, v3
           condbr v5, b1(v{slot}, v1, v2, v4), b2(v4)
         b1(v6: i32, v7: i32, v8: i32, v9: i1):
           {body}v10: i64 = pack64 v7, v8
           v11: i64 = convert zext i64 v6
           v12: i64 = int add v10, v11
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v12, v{last}, v9)
           br b2(v9)
         b2(v13: i1):
           ret"
    )
}

#[test]
fn a_costly_arm_is_skipped_when_no_lane_takes_it() {
    let lane = lane(&costly_arm());
    let packet = lowered(&lane);
    let defs = definitions(&packet);
    let answers: Vec<ValueId> = insts(&packet)
        .filter_map(|inst| match inst {
            Inst::Packet {
                op: PacketOp::Any,
                output,
                ..
            } => Some(*output),
            _ => None,
        })
        .collect();
    assert_eq!(
        answers.len(),
        1,
        "one query decides whether any lane takes the arm"
    );
    let branches_on_it = packet.blocks.values().any(|b| {
        matches!(b.term, Term::CondBr { cond, .. } if answers.contains(&cond)
            || matches!(defs[cond.0], Some(Op::Convert(_, _, a)) if answers.contains(&a)))
    });
    assert!(branches_on_it, "the packet program branches around the arm");
    let chain = |x: u32| (0..STEPS).fold(x, |y, _| y.wrapping_mul(3).wrapping_add(1));
    check(&lane, 1000, |_| None);
    check(&lane, 20, |l| (4 * l > 20).then(|| chain(4 * l)));
}

#[test]
fn an_exit_from_an_inner_loop_leaves_the_outer_loop_with_what_it_carried() {
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v59: i32 = const i32 0x2
           v60: i32 = int shl v0, v59
           v5: i32 = const i32 0x0
           br b1(v5, v5, v60, v1, v2, v4)
         b1(v6: i32, v7: i32, v8: i32, v9: i32, v10: i32, v11: i1):
           v12: i32 = const i32 0x0
           br b2(v12, v6, v7, v8, v9, v10, v11)
         b2(v13: i32, v14: i32, v15: i32, v16: i32, v17: i32, v18: i32, v19: i1):
           v20: i32 = const i32 0x1
           v21: i32 = int add v15, v20
           v22: i32 = const i32 0x4
           v23: i32 = int mul v14, v22
           v24: i32 = int add v23, v13
           v25: i32 = const i32 0x2
           v26: i32 = int add v16, v25
           v27: i1 = cmp eq v24, v26
           condbr v27, b5(v21, v14, v16, v17, v18, v19), b3(v13, v14, v21, v16, v17, v18, v19)
         b3(v28: i32, v29: i32, v30: i32, v31: i32, v32: i32, v33: i32, v34: i1):
           v35: i32 = const i32 0x1
           v36: i32 = int add v28, v35
           v37: i32 = const i32 0x4
           v38: i1 = cmp ult v36, v37
           condbr v38, b2(v36, v29, v30, v31, v32, v33, v34), b4(v29, v30, v31, v32, v33, v34)
         b4(v39: i32, v40: i32, v41: i32, v42: i32, v43: i32, v44: i1):
           v45: i32 = const i32 0x1
           v46: i32 = int add v39, v45
           br b1(v46, v40, v41, v42, v43, v44)
         b5(v47: i32, v48: i32, v49: i32, v50: i32, v51: i32, v52: i1):
           v53: i64 = pack64 v50, v51
           v54: i64 = convert zext i64 v49
           v55: i64 = int add v53, v54
           v56: i32 = const i32 0x100
           v57: i32 = int mul v48, v56
           v58: i32 = int add v47, v57
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v55, v58, v52)
           ret",
    );
    check(&lane, 0, |l| Some(260 * l + 3));
}

#[test]
fn a_lane_that_finishes_inside_a_loop_leaves_the_others_going_around() {
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v43: i32 = const i32 0x2
           v44: i32 = int shl v0, v43
           v5: i32 = const i32 0x2
           v6: i32 = int lshr v44, v5
           v7: i32 = const i32 0x0
           br b1(v7, v6, v44, v1, v2, v4)
         b1(v8: i32, v9: i32, v10: i32, v11: i32, v12: i32, v13: i1):
           v14: i1 = cmp eq v8, v9
           condbr v14, b4(v13), b2(v8, v9, v10, v11, v12, v13)
         b2(v15: i32, v16: i32, v17: i32, v18: i32, v19: i32, v20: i1):
           v21: i32 = const i32 0x3
           v22: i1 = cmp eq v15, v21
           condbr v22, b3(v15, v17, v18, v19, v20), b5(v15, v16, v17, v18, v19, v20)
         b5(v23: i32, v24: i32, v25: i32, v26: i32, v27: i32, v28: i1):
           v29: i32 = const i32 0x1
           v30: i32 = int add v23, v29
           br b1(v30, v24, v25, v26, v27, v28)
         b3(v31: i32, v32: i32, v33: i32, v34: i32, v35: i1):
           v36: i64 = pack64 v33, v34
           v37: i64 = convert zext i64 v32
           v38: i64 = int add v36, v37
           v39: i32 = const i32 0x10
           v40: i32 = int mul v31, v39
           v41: i32 = int add v32, v40
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v38, v41, v35)
           ret
         b4(v42: i1):
           ret",
    );
    check(&lane, 0, |l| (l >= 4).then(|| 4 * l + 48));
}

#[test]
fn a_bit_every_lane_agreed_on_before_a_varying_merge_is_tested_per_lane_after_it() {
    // x < 12 varies; the merge makes k 1 for some lanes and 2 for others, so
    // `k == 2` varies although both inputs of the merge are constants. Lane 0
    // is one that does not store: taking its answer for the packet's would
    // skip every lane's store.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v34: i32 = const i32 0x2
           v35: i32 = int shl v0, v34
           v5: i32 = const i32 0xc
           v6: i1 = cmp ult v35, v5
           condbr v6, b1(v35, v1, v2, v4), b2(v35, v1, v2, v4)
         b1(v7: i32, v8: i32, v9: i32, v10: i1):
           v11: i32 = const i32 0x1
           br b3(v11, v7, v8, v9, v10)
         b2(v12: i32, v13: i32, v14: i32, v15: i1):
           v16: i32 = const i32 0x2
           br b3(v16, v12, v13, v14, v15)
         b3(v17: i32, v18: i32, v19: i32, v20: i32, v21: i1):
           v22: i32 = const i32 0x2
           v23: i1 = cmp eq v17, v22
           condbr v23, b4(v18, v19, v20, v21), b5(v21)
         b4(v24: i32, v25: i32, v26: i32, v27: i1):
           v28: i64 = pack64 v25, v26
           v29: i64 = convert zext i64 v24
           v30: i64 = int add v28, v29
           v31: i32 = const i32 0x5
           v32: i32 = int add v24, v31
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v30, v32, v27)
           ret
         b5(v33: i1):
           ret",
    );
    check(&lane, 0, |l| (4 * l >= 12).then(|| 4 * l + 5));
}

#[test]
fn an_operation_that_reads_for_every_lane_reads_nothing_for_a_lane_outside_the_mask() {
    let args: Vec<String> = (0..16)
        .map(|k| match k {
            12 => "v12".to_string(),
            13 => "v13".to_string(),
            14 | 15 => "v14".to_string(),
            _ => "v7".to_string(),
        })
        .collect();
    let lane = lane(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v19: i32 = const i32 0x2
           v20: i32 = int shl v0, v19
           v5: i32 = const i32 0xc
           v6: i1 = cmp ult v20, v5
           condbr v6, b1(v20, v1, v2, v4), b2(v4)
         b1(v7: i32, v8: i32, v9: i32, v10: i1):
           v12: i32 = const i32 0x1
           v13: i1 = const i1 0x0
           v14: f32 = convert uitofp.rte f32 v7
           v15: i32 = target rdna4.image_sample_lz({}) !p1
           v16: i64 = pack64 v8, v9
           v17: i64 = convert zext i64 v7
           v18: i64 = int add v16, v17
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v18, v15, v10)
           ret
         b2(v11: i1):
           ret",
        args.join(", ")
    ));
    let packet = lowered(&lane);
    let defs = definitions(&packet);
    let sample = insts(&packet)
        .find_map(|inst| match inst {
            Inst::Target { args, .. } => Some(args.values().to_vec()),
            _ => None,
        })
        .unwrap();
    let condition = insts(&packet)
        .find_map(|inst| match inst {
            Inst::Core {
                value,
                op: Op::Cmp(IntPred::Ult, ..),
                ..
            } => Some(*value),
            _ => None,
        })
        .unwrap();
    for (index, arg) in sample.iter().enumerate() {
        let suppressed = matches!(defs[arg.0], Some(Op::Select(c, _, zero))
            if holds_under_valid(&packet, &defs, c, condition)
                && matches!(defs[zero.0], Some(Op::Const(_, 0))));
        assert_eq!(
            suppressed,
            !matches!(index, 12 | 13),
            "operand {} of the sample",
            index
        );
    }
}

#[test]
fn lanes_going_around_from_two_places_take_what_their_own_place_computed() {
    // A trip adds 1 when the trip count is odd and 2 when it is even, and both
    // places go around; each lane leaves after as many trips as its index, so
    // lanes that go around together came from different places.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v54: i32 = const i32 0x2
           v55: i32 = int shl v0, v54
           v5: i32 = const i32 0x2
           v6: i32 = int lshr v55, v5
           v7: i32 = const i32 0x0
           br b1(v7, v7, v6, v55, v1, v2, v4)
         b1(v8: i32, v9: i32, v10: i32, v11: i32, v12: i32, v13: i32, v14: i1):
           v15: i1 = cmp uge v8, v10
           condbr v15, b4(v9, v11, v12, v13, v14), b2(v8, v9, v10, v11, v12, v13, v14)
         b2(v16: i32, v17: i32, v18: i32, v19: i32, v20: i32, v21: i32, v22: i1):
           v23: i32 = const i32 0x1
           v24: i32 = int add v16, v23
           v25: i32 = int and v24, v23
           v26: i32 = const i32 0x0
           v27: i1 = cmp ne v25, v26
           condbr v27, b3(v24, v17, v18, v19, v20, v21, v22), b5(v24, v17, v18, v19, v20, v21, v22)
         b3(v28: i32, v29: i32, v30: i32, v31: i32, v32: i32, v33: i32, v34: i1):
           v35: i32 = const i32 0x1
           v36: i32 = int add v29, v35
           br b1(v28, v36, v30, v31, v32, v33, v34)
         b5(v37: i32, v38: i32, v39: i32, v40: i32, v41: i32, v42: i32, v43: i1):
           v44: i32 = const i32 0x2
           v45: i32 = int add v38, v44
           br b1(v37, v45, v39, v40, v41, v42, v43)
         b4(v46: i32, v47: i32, v48: i32, v49: i32, v50: i1):
           v51: i64 = pack64 v48, v49
           v52: i64 = convert zext i64 v47
           v53: i64 = int add v51, v52
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v53, v46, v50)
           ret",
    );
    // After n trips, the odd trips 1, 3, ... added 1 and the even ones 2.
    check(&lane, 0, |l| Some((l + 1) / 2 + 2 * (l / 2)));
}

#[test]
fn a_loop_every_lane_leaves_together_keeps_a_branch_inside_it_per_lane() {
    // Every lane goes around the same number of times, but each trip adds
    // either the lane's work item or 1 depending on the lane.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v57: i32 = const i32 0x2
           v58: i32 = int shl v0, v57
           v5: i32 = const i32 0x0
           br b1(v5, v5, v58, v3, v1, v2, v4)
         b1(v6: i32, v7: i32, v8: i32, v9: i32, v10: i32, v11: i32, v12: i1):
           v13: i1 = cmp ult v6, v9
           condbr v13, b2(v6, v7, v8, v9, v10, v11, v12), b5(v7, v8, v10, v11, v12)
         b2(v14: i32, v15: i32, v16: i32, v17: i32, v18: i32, v19: i32, v20: i1):
           v21: i32 = const i32 0x8
           v22: i1 = cmp ult v16, v21
           condbr v22, b3(v14, v15, v16, v17, v18, v19, v20), b4(v14, v15, v16, v17, v18, v19, v20)
         b3(v23: i32, v24: i32, v25: i32, v26: i32, v27: i32, v28: i32, v29: i1):
           v30: i32 = int add v24, v25
           br b6(v23, v30, v25, v26, v27, v28, v29)
         b4(v31: i32, v32: i32, v33: i32, v34: i32, v35: i32, v36: i32, v37: i1):
           v38: i32 = const i32 0x1
           v39: i32 = int add v32, v38
           br b6(v31, v39, v33, v34, v35, v36, v37)
         b6(v40: i32, v41: i32, v42: i32, v43: i32, v44: i32, v45: i32, v46: i1):
           v47: i32 = const i32 0x1
           v48: i32 = int add v40, v47
           br b1(v48, v41, v42, v43, v44, v45, v46)
         b5(v49: i32, v50: i32, v51: i32, v52: i32, v53: i1):
           v54: i64 = pack64 v51, v52
           v55: i64 = convert zext i64 v50
           v56: i64 = int add v54, v55
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v56, v49, v53)
           ret",
    );
    let packet = lowered(&lane);
    assert_eq!(queries(&packet), 0, "the trip count is still a scalar test");
    check(&lane, 6, |l| Some(if 4 * l < 8 { 6 * 4 * l } else { 6 }));
}

#[test]
fn a_store_in_a_loop_happens_on_the_trips_its_lane_makes() {
    // Each trip stores the trip count at the lane's slot; a lane makes as many
    // trips as its index, so what is left is its index and a lane that makes no
    // trip stores nothing.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v27: i32 = const i32 0x2
           v28: i32 = int shl v0, v27
           v5: i32 = const i32 0x2
           v6: i32 = int lshr v28, v5
           v7: i32 = const i32 0x0
           br b1(v7, v6, v28, v1, v2, v4)
         b1(v8: i32, v9: i32, v10: i32, v11: i32, v12: i32, v13: i1):
           v14: i1 = cmp uge v8, v9
           condbr v14, b3(v13), b2(v8, v9, v10, v11, v12, v13)
         b2(v15: i32, v16: i32, v17: i32, v18: i32, v19: i32, v20: i1):
           v21: i32 = const i32 0x1
           v22: i32 = int add v15, v21
           v23: i64 = pack64 v18, v19
           v24: i64 = convert zext i64 v17
           v25: i64 = int add v23, v24
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v25, v22, v20)
           br b1(v22, v16, v17, v18, v19, v20)
         b3(v26: i1):
           ret",
    );
    check(&lane, 0, |l| (l > 0).then_some(l));
}

#[test]
fn an_operation_with_a_lane_predicate_reads_only_for_the_lanes_in_the_mask() {
    let args: Vec<String> = (0..14)
        .map(|k| match k {
            2 => "v13".to_string(),
            13 => "v10".to_string(),
            _ => "v7".to_string(),
        })
        .collect();
    let lane = lane(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v19: i32 = const i32 0x2
           v20: i32 = int shl v0, v19
           v5: i32 = const i32 0xc
           v6: i1 = cmp ult v20, v5
           condbr v6, b1(v20, v1, v2, v4), b2(v4)
         b1(v7: i32, v8: i32, v9: i32, v10: i1):
           v12: i64 = pack64 v8, v9
           v13: i64 = convert zext i64 v7
           v14: i32, v15: i32, v16: i32, v17: i32 = target rdna4.image_bvh64_intersect_ray({}) !p1
           v18: i64 = int add v12, v13
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v18, v14, v10)
           ret
         b2(v11: i1):
           ret",
        args.join(", ")
    ));
    let packet = lowered(&lane);
    let defs = definitions(&packet);
    let predicate = insts(&packet)
        .find_map(|inst| match inst {
            Inst::Target { args, .. } => Some(args.values()[13]),
            _ => None,
        })
        .unwrap();
    let condition = insts(&packet)
        .find_map(|inst| match inst {
            Inst::Core {
                value,
                op: Op::Cmp(IntPred::Ult, ..),
                ..
            } => Some(*value),
            _ => None,
        })
        .unwrap();
    assert!(
        holds_under_valid(&packet, &defs, predicate, condition),
        "the traversal runs for the lanes the branch holds and no others"
    );
}

#[test]
fn a_record_every_lane_reads_from_the_same_address_holds_one_value_for_every_lane() {
    // Every lane but lane 0 reads eight words from the start of the buffer.
    // Kept as one value past the arm some lanes skip, the words must be the
    // words for every lane: a read that gave the lanes outside its mask zeros
    // would hand every lane lane 0's zeros.
    let body = chain(36, 42);
    let last = 42 + 4 * STEPS as usize - 1;
    let n = last + 1;
    let lane = lane(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v{two}: i32 = const i32 0x2
           v{slot}: i32 = int shl v0, v{two}
           v5: i32 = const i32 0x0
           v6: i1 = cmp ne v{slot}, v5
           condbr v6, b1(v{slot}, v1, v2, v4), b3(v4)
         b1(v7: i32, v8: i32, v9: i32, v10: i1):
           v11: i64 = pack64 v8, v9
           v12: i32 = effect !p256 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v11, v10)
           v13: i64 = const i64 0x4
           v14: i64 = int add v11, v13
           v15: i32 = effect !p257 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v14, v10)
           v16: i64 = const i64 0x8
           v17: i64 = int add v11, v16
           v18: i32 = effect !p258 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v17, v10)
           v19: i64 = const i64 0xc
           v20: i64 = int add v11, v19
           v21: i32 = effect !p259 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v20, v10)
           v22: i64 = const i64 0x10
           v23: i64 = int add v11, v22
           v24: i32 = effect !p260 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v23, v10)
           v25: i64 = const i64 0x14
           v26: i64 = int add v11, v25
           v27: i32 = effect !p261 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v26, v10)
           v28: i64 = const i64 0x18
           v29: i64 = int add v11, v28
           v30: i32 = effect !p262 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v29, v10)
           v31: i64 = const i64 0x1c
           v32: i64 = int add v11, v31
           v33: i32 = effect !p263 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v32, v10)
           v34: i32 = const i32 0xc
           v35: i1 = cmp ugt v7, v34
           condbr v35, b2(v7, v8, v9, v10, v21, v33), b4(v7, v8, v9, v10, v21, v33)
         b2(v36: i32, v37: i32, v38: i32, v39: i1, v40: i32, v41: i32):
           {body}v{a}: i64 = pack64 v37, v38
           v{b}: i64 = convert zext i64 v36
           v{c}: i64 = int add v{a}, v{b}
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v{c}, v{last}, v39)
           br b4(v36, v37, v38, v39, v40, v41)
         b4(v{x}: i32, v{lo}: i32, v{hi}: i32, v{exec}: i1, v{w3}: i32, v{w7}: i32):
           v{d}: i64 = pack64 v{lo}, v{hi}
           v{e}: i64 = convert zext i64 v{x}
           v{f}: i64 = int add v{d}, v{e}
           v{g}: i32 = int add v{w3}, v{w7}
           effect !p2 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v{f}, v{g}, v{exec})
           ret
         b3(v{rest}: i1):
           ret",
        a = n,
        b = n + 1,
        c = n + 2,
        x = n + 3,
        lo = n + 4,
        hi = n + 5,
        exec = n + 6,
        w3 = n + 7,
        w7 = n + 8,
        d = n + 9,
        e = n + 10,
        f = n + 11,
        g = n + 12,
        rest = n + 13,
        two = n + 14,
        slot = n + 15,
    ));
    check(&lane, 0, |l| {
        (l != 0).then(|| UNTOUCHED.wrapping_add(UNTOUCHED))
    });
}

#[test]
fn a_table_index_that_wraps_across_the_lanes_reads_each_lane_its_own_entry() {
    // Lane l reads entry (3 l) & 31 of a 32-entry table. The index steps by 3
    // until lane 11 wraps it back to 1, so reading the lanes' entries as one
    // run of records 12 bytes apart would read past the table.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v5: i32 = const i32 0x3
           v6: i32 = int mul v0, v5
           v7: i32 = const i32 0x1f
           v8: i32 = int and v6, v7
           v9: i64 = convert zext i64 v8
           v10: i64 = const i64 0x2
           v11: i64 = int shl v9, v10
           v12: i64 = const i64 0x80
           v13: i64 = int add v11, v12
           v14: i64 = pack64 v1, v2
           v15: i64 = int add v14, v13
           v16: i1 = const i1 0x1
           v17: i32 = effect !p256 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v15, v16)
           v18: i32 = const i32 0x2
           v19: i32 = int shl v0, v18
           v20: i64 = convert zext i64 v19
           v21: i64 = int add v14, v20
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v21, v17, v4)
           ret",
    );
    let table = |slot: usize| {
        if slot >= 32 {
            1000 + slot as u32 - 32
        } else {
            UNTOUCHED
        }
    };
    // Room past the table for whatever a wrong read would reach.
    check_over(&lane, 0, 128, table, |l| Some(1000 + (3 * l) % 32));
}

#[test]
fn a_shift_by_the_whole_word_leaves_each_lane_its_own_entry() {
    // A shift of a 32-bit word by 32 shifts it by 0, so lane l reads entry l.
    let lane = lane(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i1):
           v5: i32 = const i32 0x20
           v6: i32 = int lshr v0, v5
           v7: i64 = convert zext i64 v6
           v8: i64 = const i64 0x2
           v9: i64 = int shl v7, v8
           v10: i64 = const i64 0x80
           v11: i64 = int add v9, v10
           v12: i64 = pack64 v1, v2
           v13: i64 = int add v12, v11
           v14: i1 = const i1 0x1
           v15: i32 = effect !p256 memory load.b32 global cu relaxed temporal volatile=0 deferred=0 (v13, v14)
           v16: i32 = const i32 0x2
           v17: i32 = int shl v0, v16
           v18: i64 = convert zext i64 v17
           v19: i64 = int add v12, v18
           effect !p0 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v19, v15, v4)
           ret",
    );
    let table = |slot: usize| {
        if slot >= 32 {
            1000 + slot as u32 - 32
        } else {
            UNTOUCHED
        }
    };
    check_over(&lane, 0, 128, table, |l| Some(1000 + l));
}

/// A lane program dispatched as a wave: one workgroup of `count` work items,
/// run alone and in packets of every width, with `shared` in SGPR 0 (where
/// the dispatch puts the packet's address) and the output buffer's address
/// in SGPRs 2 and 3 (where it puts the kernarg segment's).
fn wave(text: &str) -> LiftedFunction {
    let registry = crate::rdna_spmd::targets::rdna4::registry();
    let ir = parse::func(&registry, text).unwrap();
    let input = |source, ty| Parameter { source, ty };
    LiftedFunction {
        registry: Arc::new(registry),
        ir,
        parameter_inputs: vec![
            input(ParameterSource::Vgpr(0), Ty::I32),
            input(ParameterSource::Sgpr(0), Ty::I32),
            input(ParameterSource::Sgpr(1), Ty::I32),
            input(ParameterSource::Sgpr(2), Ty::I32),
            input(ParameterSource::Sgpr(3), Ty::I32),
            input(ParameterSource::MaskBit(126), Ty::I1),
        ],
        revision: 0,
    }
}

fn check_wave(
    lane: &LiftedFunction,
    count: u32,
    shared: u32,
    expected: impl Fn(u32) -> Option<u32>,
) {
    use crate::rdna_spmd::compiler::compile_lane;
    use crate::rdna_spmd::engine::kernel::Kernel;
    use crate::rdna_spmd::{dispatch, GridDims};
    let mut kd = crate::processor::decode_kernel_desc(&[0; 64]);
    kd.enable_sgpr_dispatch_ptr = true;
    kd.enable_sgpr_kernarg_segment_ptr = true;
    let dims = GridDims {
        num_wg_x: 1,
        num_wg_y: 1,
        num_wg_z: 1,
        wg_x: count,
        wg_y: 1,
        wg_z: 1,
    };
    for width in [0u32, 1, 2, 4, 8, 16, 32] {
        // A packet that holds the whole wave runs alone, and packets that run
        // alone must hold whole packets of the workgroup.
        if width == 32 && count % width != 0 {
            continue;
        }
        let mut output = vec![UNTOUCHED; count as usize];
        let alone = super::super::Lane::from(lane.clone());
        let (code, scheduler) = compile_lane(&alone, 256, width, Some(count))
            .expect("every lane is at each operation over the wave");
        let kernel = Kernel::new(code, scheduler, width);
        dispatch(
            &kernel,
            &kd,
            output.as_mut_ptr() as u64,
            shared as u64,
            dims,
            0,
            0,
            1,
        );
        let wanted: Vec<u32> = (0..count)
            .map(|l| expected(l).unwrap_or(UNTOUCHED))
            .collect();
        assert_eq!(output, wanted, "width {} with {} lanes", width, count);
    }
}

/// The slot of the lane's own word in the output, from the address in SGPRs
/// 2 and 3, as `v10`.
const SLOT: &str = "v6: i32 = const i32 0x2
                    v7: i32 = int shl v0, v6
                    v8: i64 = convert zext i64 v7
                    v9: i64 = pack64 v3, v4
                    v10: i64 = int add v9, v8";

#[test]
fn a_query_kept_in_an_arm_asks_every_lane_of_the_wave_at_the_arm() {
    // Lanes below 16 take the arm; there, whether the lane `shared` names is
    // among them is asked of the wave, so packets holding no lane of the arm
    // still meet the query, and a lane the arm leaves out is not counted.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x10
           v12: i1 = cmp ult v0, v11
           condbr v12, b1(v0, v1, v10, v5), b2(v5)
         b1(v13: i32, v14: i32, v15: i64, v16: i1):
           v17: i1 = cmp eq v13, v14
           v18: i1 = int and v17, v16
           v19: i1 = effect !p0 wave any (v18)
           v20: i1 = int and v19, v16
           v21: i32 = const i32 0x1
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v15, v21, v20)
           ret
         b2(v22: i1):
           ret"
    ));
    check_wave(&lane, 32, 3, |l| (l < 16).then_some(1));
    check_wave(&lane, 32, 20, |_| None);
    check_wave(&lane, 20, 15, |l| (l < 16).then_some(1));
}

#[test]
fn a_ballot_kept_holds_the_bits_of_every_lane_of_the_wave() {
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x1
           v12: i32 = int and v0, v11
           v13: i1 = cmp eq v12, v11
           v14: i1 = int and v13, v5
           v15: i32 = effect !p0 wave ballot (v14)
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v10, v15, v5)
           ret"
    ));
    check_wave(&lane, 32, 0, |_| Some(0xaaaa_aaaa));
    check_wave(&lane, 20, 0, |_| Some(0x000a_aaaa));
}

#[test]
fn a_read_of_the_first_lane_kept_reads_the_first_lane_of_the_wave() {
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x3
           v12: i32 = int mul v0, v11
           v13: i32 = int add v12, v1
           v14: i32 = effect !p0 wave readfirstlane (v13, v5)
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v10, v14, v5)
           ret"
    ));
    check_wave(&lane, 32, 7, |_| Some(7));
    check_wave(&lane, 20, 100, |_| Some(100));
}

#[test]
fn a_loop_holding_a_query_kept_goes_around_while_any_lane_of_the_wave_does() {
    // Each lane counts down from its index; the loop goes around while any
    // lane of the wave has a count left, and every lane stores the trips
    // the wave made: one more than the highest index, for the trip that
    // finds no lane left.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x0
           br b1(v0, v11, v10, v5)
         b1(v12: i32, v13: i32, v14: i64, v15: i1):
           v16: i32 = const i32 0x0
           v17: i1 = cmp ugt v12, v16
           v18: i32 = const i32 0x1
           v19: i32 = int sub v12, v18
           v20: i32 = select v17, v19, v12
           v21: i32 = int add v13, v18
           v22: i1 = int and v17, v15
           v23: i1 = effect !p0 wave any (v22)
           condbr v23, b1(v20, v21, v14, v15), b2(v21, v14, v15)
         b2(v24: i32, v25: i64, v26: i1):
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v25, v24, v26)
           ret"
    ));
    check_wave(&lane, 32, 0, |_| Some(32));
    check_wave(&lane, 20, 0, |_| Some(20));
}

#[test]
fn a_read_of_a_lane_kept_reads_that_lane_of_the_wave() {
    // The lane `shared` names is read by every lane, whichever packet holds
    // it.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x3
           v12: i32 = int mul v0, v11
           v13: i32 = const i32 0x64
           v14: i32 = int add v12, v13
           v15: i32 = const i32 0x0
           v16: i32 = effect !p0 wave readlane (v14, v1, v15)
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v10, v16, v5)
           ret"
    ));
    check_wave(&lane, 32, 17, |_| Some(3 * 17 + 100));
    check_wave(&lane, 20, 19, |_| Some(3 * 19 + 100));
}

#[test]
fn a_write_into_a_lane_kept_is_read_back_from_that_lane() {
    // A value stashed in lane 5 of register 3 is read back by every lane.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x309
           v12: i32 = const i32 0x5
           v13: i32 = const i32 0x3
           v14: i32 = effect !p0 wave writelane (v11, v12, v0, v13)
           v15: i32 = effect !p1 wave readlane (v14, v12, v13)
           effect !p2 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v10, v15, v5)
           ret"
    ));
    check_wave(&lane, 32, 0, |_| Some(0x309));
}

#[test]
fn an_exchange_where_lanes_may_be_elsewhere_is_refused() {
    use crate::rdna_spmd::compiler::compile_lane;
    // Lanes below 16 read a lane of the wave while the others have left.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x10
           v12: i1 = cmp ult v0, v11
           condbr v12, b1(v0, v1, v10, v5), b2(v5)
         b1(v13: i32, v14: i32, v15: i64, v16: i1):
           v17: i32 = const i32 0x0
           v18: i32 = effect !p0 wave readlane (v13, v14, v17)
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v15, v18, v16)
           ret
         b2(v19: i1):
           ret"
    ));
    for width in [0u32, 8, 32] {
        let refusal = compile_lane(&super::super::Lane::from(lane.clone()), 256, width, None)
            .err()
            .expect("the read needs every lane at it");
        assert_eq!(
            refusal.reason, "an operation over every lane where lanes may be elsewhere",
            "width {width}"
        );
    }
}

#[test]
fn a_wave_operation_in_a_span_every_lane_takes_or_leaves_runs_only_when_taken() {
    // Whether `shared` is set decides for every lane whether the read of
    // the lane it names happens, so the packets branch together and the
    // read, kept, runs with every lane at it.
    let lane = wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x0
           v12: i1 = cmp ne v1, v11
           condbr v12, b1(v0, v1, v10, v5), b2(v5)
         b1(v13: i32, v16: i32, v14: i64, v15: i1):
           v17: i32 = const i32 0x0
           v18: i32 = effect !p0 wave readlane (v13, v16, v17)
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v14, v18, v15)
           ret
         b2(v19: i1):
           ret"
    ));
    check_wave(&lane, 32, 5, |_| Some(5));
    check_wave(&lane, 32, 0, |_| None);
}

/// A loop of `trips` trips summing what the lane `shared` names holds each
/// trip, where each lane holds ten times its index plus the trip.
fn summing_a_lane_over(trips: &str) -> LiftedFunction {
    wave(&format!(
        "func entry b0
         b0(v0: i32, v1: i32, v2: i32, v3: i32, v4: i32, v5: i1):
           {SLOT}
           v11: i32 = const i32 0x0
           br b1(v0, v11, v11, {trips}, v1, v10, v5)
         b1(v12: i32, v13: i32, v14: i32, v15: i32, v21: i32, v16: i64, v17: i1):
           v18: i32 = const i32 0xa
           v19: i32 = int mul v12, v18
           v20: i32 = int add v19, v13
           v22: i32 = const i32 0x0
           v23: i32 = effect !p0 wave readlane (v20, v21, v22)
           v24: i32 = int add v14, v23
           v25: i32 = const i32 0x1
           v26: i32 = int add v13, v25
           v27: i1 = cmp ult v26, v15
           condbr v27, b1(v12, v26, v24, v15, v21, v16, v17), b2(v24, v16, v17)
         b2(v28: i32, v29: i64, v30: i1):
           effect !p1 memory store.b32 global cu relaxed temporal volatile=0 deferred=0 (v29, v28, v30)
           ret"
    ))
}

#[test]
fn an_exchange_in_a_loop_the_lanes_go_around_together_reads_the_wave_each_trip() {
    let lane = summing_a_lane_over("v1");
    check_wave(&lane, 32, 4, |_| Some(40 + 41 + 42 + 43));
}

#[test]
fn an_exchange_in_a_loop_the_lanes_leave_apart_is_refused() {
    use crate::rdna_spmd::compiler::compile_lane;
    // Each lane makes as many trips as its index, so lanes leave the loop
    // while others still read lane 3 in it.
    let lane = summing_a_lane_over("v0");
    let refusal = compile_lane(&super::super::Lane::from(lane), 256, 8, None)
        .err()
        .expect("lanes leave the loop apart");
    assert_eq!(
        refusal.reason,
        "an operation over every lane in a loop the lanes leave apart"
    );
}
