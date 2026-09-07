use super::*;
use crate::rdna_instructions::{VOP3,VIMAGE};

fn program(inst:InstFormat)->ScalarProgram {
    assert!(matches!(instruction(&inst),Lowering::TypedAlu {..}));
    ScalarProgram {entry_pc:0,blocks:BTreeMap::from([(0,ScalarBlock {pc:0,body:vec![
        InstFormat::SOP1(SOP1 {op:I::S_MOV_B32,sdst:126,ssrc0:SourceOperand::ScalarRegister(20)}),inst,
    ],term:Terminator::Return})])}
}

#[test]
fn f64_fixup_preserves_aliases_special_values_and_inactive_lanes() {
    let p=program(InstFormat::VOP3(VOP3 {op:I::V_DIV_FIXUP_F64,vdst:4,
        src0:SourceOperand::FloatConstant(123.0),src1:SourceOperand::VectorRegister(4),src2:SourceOperand::VectorRegister(6),
        abs:0,neg:0,opsel:0,cm:0,omod:0}));
    let cases=[
        (2f64.to_bits(),6f64.to_bits(),3f64.to_bits()),
        (0,0,0xfff8_0000_0000_0000),
        (2f64.to_bits(),0x8000_0000_0000_0000,0x8000_0000_0000_0000),
        (f64::INFINITY.to_bits(),f64::INFINITY.to_bits(),0xfff8_0000_0000_0000),
        (0x7ff0_0000_0000_0012,0xfff0_0000_0000_0034,0xfff8_0000_0000_0034),
        (f64::MAX.to_bits(),f64::MIN_POSITIVE.to_bits(),0),
    ];
    for width in [0,1,2,4,8,16] {
        let w=width.max(1) as usize;
        for &(den,num,result) in &cases {
            for mask in [0u32,0xaaaa_aaaa,u32::MAX] {
                let mut s=[0u32;crate::rdna_spmd::emit::COOP_SGPR_BUF];s[20]=mask;
                let mut v=vec![0u32;256*w];
                for lane in 0..w {v[4*w+lane]=den as u32;v[5*w+lane]=(den>>32) as u32;v[6*w+lane]=num as u32;v[7*w+lane]=(num>>32) as u32;}
                run_memory_case(&p,width,&mut s,&mut v,0,0,0);
                for lane in 0..w {assert_eq!(v[4*w+lane] as u64|((v[5*w+lane] as u64)<<32),if mask>>lane&1!=0 {result}else{den},"width={width} lane={lane} mask={mask:x}");}
            }
        }
    }
}

#[test]
fn bvh_target_preserves_four_results_aliases_and_packet_masks() {
    // Two equal box nodes permit both uniform-node and divergent-node paths.
    let mut storage=vec![0u32;128];
    let aligned=(storage.as_mut_ptr() as usize+255)&!255;
    let start=(aligned-storage.as_ptr() as usize)/4;
    let mut node=vec![0x100,0x200,0x300,0x400];
    for bounds in [[3f32,-1.,-1.,4.,1.,1.],[1.,-1.,-1.,2.,1.,1.],[5.,-1.,-1.,6.,1.,1.],[-2.,-1.,-1.,-1.,1.,1.]] {node.extend(bounds.iter().map(|x|x.to_bits()));}
    node.extend([0,0,4]);node.resize(32,0);
    storage[start..start+32].copy_from_slice(&node);storage[start+32..start+64].copy_from_slice(&node);
    let address=aligned as u64;
    for dst in [4,16] {
        let p=program(InstFormat::VIMAGE(VIMAGE {op:I::IMAGE_BVH64_INTERSECT_RAY,dim:0,r128:0,d16:0,a16:0,dmask:15,vdata:dst,rsrc:0,scope:0,th:0,tfe:0,vaddr0:0,vaddr1:2,vaddr2:3,vaddr3:6,vaddr4:9}));
        for width in [0,1,2,4,8,16] {
            let w=width.max(1) as usize;
            for (divergent,sorted) in [(false,true),(true,true),(false,false)] {
                for mask in [0u32,0xaaaa_aaaa,u32::MAX] {
                    let mut s=[0u32;crate::rdna_spmd::emit::COOP_SGPR_BUF];s[0]=(address>>8) as u32;s[1]=(address>>40) as u32|if sorted {0x8000_0000}else{0};s[20]=mask;
                    let mut v=vec![0xdead_beef;256*w];
                    for lane in 0..w {
                        let ray=[5+if divergent && lane%2==1 {16}else{0},0,100f32.to_bits(),0,0,0,1f32.to_bits(),0,0,1f32.to_bits(),f32::INFINITY.to_bits(),f32::INFINITY.to_bits()];
                        for (r,value) in ray.iter().enumerate() {v[r*w+lane]=*value;}
                    }
                    let before=v.clone();run_memory_case(&p,width,&mut s,&mut v,0,0,0);
                    let expected=if sorted {[0x200,0x100,0x300,u32::MAX]}else{[0x100,0x200,0x300,u32::MAX]};
                    for lane in 0..w {for k in 0..4 {let index=(dst as usize+k)*w+lane;assert_eq!(v[index],if mask>>lane&1!=0 {expected[k]}else{before[index]},"width={width} dst={dst} lane={lane} mask={mask:x} divergent={divergent} sorted={sorted}");}}
                }
            }
        }
    }
}

#[test]
fn scratch_environment_operands_preserve_native_word_views() {
    let p = ScalarProgram { entry_pc: 0, blocks: BTreeMap::from([(0, ScalarBlock {
        pc: 0, body: vec![
            InstFormat::VOP1(crate::rdna_instructions::VOP1 { op: I::V_MOV_B32,
                src0: SourceOperand::PrivateBase, vdst: 4 }),
            InstFormat::SOP1(SOP1 { op: I::S_MOV_B64,
                ssrc0: SourceOperand::PrivateBase, sdst: 22 }),
        ], term: Terminator::Return,
    })]) };
    let base = 0x1234_abcd_7654_3210u64;
    for width in [0, 1, 2, 4, 8, 16] {
        let w = width.max(1) as usize;
        let mut s = [0u32; crate::rdna_spmd::emit::COOP_SGPR_BUF];
        s[126] = u32::MAX;
        let mut v = vec![0u32; 32 * w];
        run_memory_case(&p, width, &mut s, &mut v, base, 0, 0);
        assert_eq!(s[22] as u64 | ((s[23] as u64) << 32), base);
        assert_eq!(&v[4*w..5*w], vec![base as u32; w]);
    }
}
