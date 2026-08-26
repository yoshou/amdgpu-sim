//! VIMAGE, the image format that takes no sampler: the ray-tracing
//! instructions.
//!
//! These read what they work on from memory, so each case states the node the
//! harness puts in its data buffer as well as the ray. The address of a node
//! is split between the resource's base address and the node pointer, and the
//! harness builds both: `origin` says how much of the buffer is behind the
//! base, in the 256-byte units a base address is given in, and the pointer
//! carries the rest in eight-byte ones.
//!
//! The results are the ten VGPRs image_bvh8_intersect_ray writes;
//! image_bvh64_intersect_ray writes the first four and the rest stay clear, so
//! a case that wrote further than its instruction should have fails.

use crate::compare::*;
use crate::encoding::*;
use crate::harness::*;
use amdgpu_sim::rdna_processor::Engine;

/// The BVH resource for image_bvh64_intersect_ray, as Table 65 lays it out:
/// box sorting on with the closest-first heuristic and no box growth, a size
/// covering the whole address space, and the triangle return mode that gives
/// barycentrics rather than a triangle ID. The base address is left clear for
/// the harness to fold in.
const BVH64_RSRC: [u32; 4] = [0, 0x8000_0000, 0xFFFF_FFFF, 0x8100_03FF];

/// The same resource asking for the children that point at triangle nodes to
/// be sorted before the ones that point at boxes.
const BVH64_RSRC_TRIANGLES_FIRST: [u32; 4] = [0, 0x8010_0000, 0xFFFF_FFFF, 0x8100_03FF];

/// The same resource with box sorting off, which leaves the children in the
/// order the node holds them.
const BVH64_RSRC_UNSORTED: [u32; 4] = [0, 0x0000_0000, 0xFFFF_FFFF, 0x8100_03FF];

/// The eight-wide resource with box sorting off.
const BVH8_RSRC_UNSORTED: [u32; 4] = [0, 0x0010_0000, 0xFFFF_FFFF, 0x8168_03FF];

/// The resource for image_bvh8_intersect_ray, which asks for more: triangles
/// sorted before boxes, sorting across all eight children, and bit 115, which
/// Table 65 reserves. The part answers a node with nothing at all unless the
/// first is set, and misses every child unless the last is.
const BVH8_RSRC: [u32; 4] = [0, 0x8010_0000, 0xFFFF_FFFF, 0x8168_03FF];

/// One ray against one node.
pub(crate) struct VimageCase {
    /// What the data buffer holds: the node sits at its start.
    node: Vec<u32>,
    /// v0..v11: where the node is and what type it is, then the ray, laid out
    /// as the instruction's address groups want it.
    ray: [u32; 12],
    /// v12..v21 after the instruction.
    expected: [u32; 10],
}

fn bits(value: f32) -> u32 {
    value.to_bits()
}

/// A ray for image_bvh64_intersect_ray: VADDR0 is the node pointer pair,
/// VADDR1 the extent, VADDR2 the origin, VADDR3 the direction and VADDR4 its
/// reciprocal, which the box test uses instead of dividing.
fn ray64(node_type: u32, extent: f32, origin: [f32; 3], dir: [f32; 3]) -> [u32; 12] {
    [
        0,
        node_type,
        bits(extent),
        bits(origin[0]),
        bits(origin[1]),
        bits(origin[2]),
        bits(dir[0]),
        bits(dir[1]),
        bits(dir[2]),
        bits(1.0 / dir[0]),
        bits(1.0 / dir[1]),
        bits(1.0 / dir[2]),
    ]
}

/// A ray for image_bvh8_intersect_ray, whose groups differ: VADDR0 is the BVH
/// base, VADDR1 the extent and the instance mask, VADDR2 the origin, VADDR3 the
/// direction, and VADDR4 the offset of the node within the BVH -- in eight-byte
/// units, with the node type in its low four bits.
fn ray8(node_index: u32, extent: f32, mask: u32, origin: [f32; 3], dir: [f32; 3]) -> [u32; 12] {
    [
        0,
        0,
        bits(extent),
        mask,
        bits(origin[0]),
        bits(origin[1]),
        bits(origin[2]),
        bits(dir[0]),
        bits(dir[1]),
        bits(dir[2]),
        node_index,
        0,
    ]
}

/// The type-5 node image_bvh64_intersect_ray tests against: four child
/// pointers, then a box for each of them, then the words the builder keeps for
/// itself.
fn box4_node(children: [u32; 4], boxes: [[f32; 6]; 4]) -> Vec<u32> {
    let mut node = children.to_vec();
    for b in boxes {
        node.extend(b.iter().map(|&x| bits(x)));
    }
    node.extend([0, 0, children.len() as u32]);
    node.resize(32, 0);
    node
}

/// The node types 0 and 1 name: a pair of triangles sharing an edge, given as
/// four vertices. Type 0 is (v0, v1, v2) and type 1 is (v3, v2, v1). `flags`
/// says which barycentric each of the last two results holds, in two bits
/// apiece, and the pair's two triangles take a byte of it each.
fn tri_pair_node(vertices: [[f32; 3]; 4], prim: [u32; 2], flags: u32) -> Vec<u32> {
    let mut node = Vec::new();
    for v in vertices {
        node.extend(v.iter().map(|&x| bits(x)));
    }
    node.extend([0, prim[0], prim[1], flags]);
    node.resize(32, 0);
    node
}

/// One child of an eight-wide box node: its box quantized to twelve bits an
/// axis, the type of the node it points at, how many nodes that child covers,
/// and the instance mask that can cull it.
struct Box8Child {
    min: [u32; 3],
    max: [u32; 3],
    node_type: u32,
    range: u32,
    mask: u32,
}

/// The type-5 node image_bvh8_intersect_ray tests against. The children's
/// boxes are quantized around an origin, one exponent per axis: a child's
/// bound is origin + q * 2^(exponent - 139). The two base addresses are where
/// the child pointers count from -- box children from the first and everything
/// else from the second -- and each child's range moves the next child of its
/// kind further along.
fn box8_node(
    box_base: u32,
    prim_base: u32,
    origin: [f32; 3],
    exponent: [u32; 3],
    children: &[Box8Child],
) -> Vec<u32> {
    let mut node = vec![
        box_base,
        prim_base,
        0xFFFF_FFFF,
        bits(origin[0]),
        bits(origin[1]),
        bits(origin[2]),
        exponent[0]
            | (exponent[1] << 8)
            | (exponent[2] << 16)
            | (6 << 24)
            | ((children.len() as u32 - 1) << 28),
        0x7F,
    ];
    for child in children {
        node.push((child.min[0] & 0xFFF) | ((child.min[1] & 0xFFF) << 12));
        node.push((child.min[2] & 0xFFF) | ((child.max[0] & 0xFFF) << 12) | (child.mask << 24));
        node.push(
            (child.max[1] & 0xFFF)
                | ((child.max[2] & 0xFFF) << 12)
                | (child.node_type << 24)
                | (child.range << 28),
        );
    }
    node.resize(32, 0);
    node
}

/// A bit-addressed node under construction: the packet format packs its fields
/// without regard for dword boundaries.
struct NodeBits {
    words: Vec<u32>,
}

impl NodeBits {
    fn new() -> Self {
        NodeBits { words: vec![0; 32] }
    }

    fn put(&mut self, position: usize, size: usize, value: u32) {
        for i in 0..size {
            if (value >> i) & 1 == 1 {
                self.words[(position + i) / 32] |= 1 << ((position + i) % 32);
            }
        }
    }
}

/// One pair of triangles in a packet: three vertex indices for each triangle,
/// and whether the pair ends the range of triangles this node covers. A pair
/// can hold a single triangle, which leaves the second one degenerate.
struct PacketPair {
    first: [u32; 3],
    second: Option<[u32; 3]>,
    range_end: bool,
}

/// The node types 0..3 and 8..11 name, one type per pair: a table of vertices,
/// a descriptor per pair naming three of them for each triangle, and the
/// primitive indices, which are compressed -- the first triangle's index is
/// given in full and the rest hold only their low bits.
fn packet_node(
    vertices: &[[f32; 3]],
    pairs: &[PacketPair],
    anchor_size: usize,
    payload_size: usize,
    midpoint: usize,
    prims: &[u32],
) -> Vec<u32> {
    let mut node = NodeBits::new();
    // The first three fields say that each vertex component is a whole 32-bit
    // float, the next that none of their low bits were dropped, and the last
    // how many vertices the table holds.
    node.put(0, 5, 31);
    node.put(5, 5, 31);
    node.put(10, 5, 31);
    node.put(20, 8, vertices.len() as u32 - 1);
    node.put(28, 3, pairs.len() as u32 - 1);
    node.put(32, 5, anchor_size as u32);
    node.put(37, 5, payload_size as u32);
    node.put(42, 10, midpoint as u32);
    for (i, vertex) in vertices.iter().enumerate() {
        for (k, component) in vertex.iter().enumerate() {
            node.put(52 + 96 * i + 32 * k, 32, bits(*component));
        }
    }
    // The descriptors are packed downwards from the end of the node.
    for (i, pair) in pairs.iter().enumerate() {
        let indices = |t: [u32; 3]| (t[0] | (t[1] << 4) | (t[2] << 8)) & 0xFFF;
        let mut descriptor = pair.range_end as u32;
        if let Some(second) = pair.second {
            descriptor |= indices(second) << 3;
        }
        descriptor |= indices(pair.first) << 17;
        node.put(1024 - (i + 1) * 29, 29, descriptor);
    }
    let mut position = midpoint;
    node.put(position, anchor_size, prims[0]);
    position += anchor_size;
    for &prim in &prims[1..] {
        node.put(position, payload_size, prim);
        position += payload_size;
    }
    node.words
}

/// Bit-exact comparison of one BVH instruction against captured hardware.
fn check_vimage(words: &[u32], rsrc: [u32; 4], origin: u32, cases: &[VimageCase]) {
    let harness = Harness::vimage();
    let mut uni = vec![0u32; 8];
    uni[..4].copy_from_slice(&rsrc);
    uni[4] = origin;

    let mut failures = Vec::new();
    for (i, case) in cases.iter().enumerate() {
        let mut src = vec![0u32; LANES * harness.src_stride];
        for lane in 0..LANES {
            src[lane * harness.src_stride..(lane + 1) * harness.src_stride]
                .copy_from_slice(&case.ray);
        }
        for engine in [Engine::Interpreter, Engine::LlvmJit] {
            let out = harness.run_with_data(engine, words, &src, &uni, &case.node);
            let got: Vec<u32> = out[..10].to_vec();
            if got == case.expected {
                continue;
            }
            let show = |values: &[u32]| {
                values
                    .iter()
                    .map(|&v| show_f32(v))
                    .collect::<Vec<String>>()
                    .join(", ")
            };
            failures.push(format!(
                "  {:<11} case {} hardware=({}) simulator=({})",
                engine_name(engine),
                i,
                show(&case.expected),
                show(&got),
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} case-results differ from hardware:\n{}",
        failures.len(),
        cases.len() * 2,
        failures.join("\n"),
    );
}

/// image_bvh64_intersect_ray: VDATA is v12, the resource s[12:15], and the
/// address groups are the ones the harness fills.
fn bvh64(cases: &[VimageCase]) {
    bvh64_with(BVH64_RSRC, 0, cases);
}

/// As `bvh64`, but against a resource of the case's own, based `origin`
/// 256-byte units into the data buffer.
fn bvh64_with(rsrc: [u32; 4], origin: u32, cases: &[VimageCase]) {
    check_vimage(
        &vimage(26, 0, 1, 0xF, 12, 12, [0, 2, 3, 6, 9]),
        rsrc,
        origin,
        cases,
    );
}

fn bvh8(cases: &[VimageCase]) {
    bvh8_with(BVH8_RSRC, cases);
}

/// As `bvh8`, but against a resource of the case's own.
fn bvh8_with(rsrc: [u32; 4], cases: &[VimageCase]) {
    check_vimage(
        &vimage(129, 0, 1, 0xF, 12, 12, [0, 2, 4, 7, 10]),
        rsrc,
        0,
        cases,
    );
}

/// Four boxes in a row along x, the last of them behind the origin.
fn box4() -> Vec<u32> {
    box4_node(
        [0x100, 0x200, 0x300, 0x400],
        [
            [3.0, -1.0, -1.0, 4.0, 1.0, 1.0],
            [1.0, -1.0, -1.0, 2.0, 1.0, 1.0],
            [5.0, -1.0, -1.0, 6.0, 1.0, 1.0],
            [-2.0, -1.0, -1.0, -1.0, 1.0, 1.0],
        ],
    )
}

/// The same four boxes with child pointers of the case's own: the low three
/// bits of a pointer are the type of the node it points at, which is what
/// decides where the child sorts when triangles come first.
fn box4_types(types: [u32; 4]) -> Vec<u32> {
    let mut children = [0u32; 4];
    for (i, child) in children.iter_mut().enumerate() {
        *child = (0x10 * (i as u32 + 1)) | types[i];
    }
    box4_node(
        children,
        [
            [1.0, -1.0, -1.0, 2.0, 1.0, 1.0],
            [3.0, -1.0, -1.0, 4.0, 1.0, 1.0],
            [5.0, -1.0, -1.0, 6.0, 1.0, 1.0],
            [7.0, -1.0, -1.0, 8.0, 1.0, 1.0],
        ],
    )
}

/// The unit square in the plane z = 1, as a pair of triangles.
const SQUARE: [[f32; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

/// The same square wound the other way round, which flips the sign of the
/// denominator the instruction returns.
const SQUARE_CW: [[f32; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

/// Eight boxes in a row along x, quantized with an exponent of 139, which
/// makes a quantized unit one world unit.
fn box8(range: u32, mask: impl Fn(usize) -> u32) -> Vec<u32> {
    let children: Vec<Box8Child> = (0..8)
        .map(|i| Box8Child {
            min: [2 * i as u32 + 1, 0, 0],
            max: [2 * i as u32 + 2, 100, 100],
            node_type: 5,
            range,
            mask: mask(i),
        })
        .collect();
    box8_node(0x10, 0x20, [0.0, 0.0, 0.0], [139, 139, 139], &children)
}

/// Two triangles in the plane z = 1 as the first pair, and the second of them
/// again as a second pair that ends the range.
fn packet() -> Vec<u32> {
    packet_node(
        &[
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
        ],
        &[
            PacketPair { first: [0, 1, 2], second: Some([3, 4, 5]), range_end: false },
            PacketPair { first: [3, 4, 5], second: None, range_end: true },
        ],
        8,
        8,
        900,
        &[55, 33, 40, 41],
    )
}

#[test]
fn image_bvh64_intersect_ray_box() {
    // A box node returns its children's pointers in the order the ray reaches
    // them, with 0xFFFFFFFF where a child was not hit at all.
    bvh64(&[
        VimageCase {
            node: box4(),
            ray: ray64(5, 100.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            expected: [0x0000_0200, 0x0000_0100, 0x0000_0300, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        }, // three of the four children hit, in intersection order
        VimageCase {
            node: box4(),
            ray: ray64(5, 4.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
            expected: [0x0000_0200, 0x0000_0100, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        }, // the ray extent clips the farthest child
        VimageCase {
            node: box4(),
            ray: ray64(5, 100.0, [1.5, 0.0, 0.0], [1.0, 0.0, 0.0]),
            expected: [0x0000_0200, 0x0000_0100, 0x0000_0300, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        }, // the ray starts inside the second child
        VimageCase {
            node: box4(),
            ray: ray64(5, 100.0, [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]),
            expected: [0x0000_0400, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        }, // the ray points the other way, so only the child behind it is hit
        VimageCase {
            node: box4(),
            ray: ray64(5, 100.0, [0.0, 5.0, 0.0], [1.0, 0.0, 0.0]),
            expected: [0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        }, // the ray passes over every child
    ]);
}

#[test]
fn image_bvh64_intersect_ray_triangle() {
    // A triangle node returns the intersection time as a fraction -- numerator
    // then denominator -- and two of the three barycentrics, which the node's
    // flags choose. A miss is an infinite numerator, signed so that the
    // fraction is positive infinity either way.
    //
    // The vertices and the ray origins are exact in binary, because the part
    // moves the triangle to the ray origin before it starts and the simulator
    // does not: the two agree to the last bit only where neither has anything
    // to round.
    bvh64(&[
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x3F80_0000, 0x3F80_0000, 0x3F00_0000, 0x3E80_0000, 0, 0, 0, 0, 0, 0],
        }, // the first triangle of the pair, hit
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(1, 100.0, [0.75, 0.75, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x3F80_0000, 0x3F80_0000, 0x3F00_0000, 0x3E80_0000, 0, 0, 0, 0, 0, 0],
        }, // the second triangle of the pair, hit
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(1, 100.0, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x7F80_0000, 0x3F80_0000, 0xBF00_0000, 0x3F40_0000, 0, 0, 0, 0, 0, 0],
        }, // missed on the other side of the shared edge
        VimageCase {
            node: tri_pair_node(SQUARE_CW, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.75, 0.75, 0.0], [0.0, 0.0, 1.0]),
            expected: [0xFF80_0000, 0xBF80_0000, 0x3F00_0000, 0xBF40_0000, 0, 0, 0, 0, 0, 0],
        }, // a miss with the other winding: both halves of the fraction negate
        VimageCase {
            node: tri_pair_node(SQUARE_CW, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [0xBF80_0000, 0xBF80_0000, 0xBF00_0000, 0xBE80_0000, 0, 0, 0, 0, 0, 0],
        }, // a hit with the other winding
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.25, 0.25, 0.0], [1.0, 0.0, 0.0]),
            expected: [0x7F80_0000, 0x0000_0000, 0xBF80_0000, 0x3F80_0000, 0, 0, 0, 0, 0, 0],
        }, // the ray runs parallel to the plane: the denominator is zero
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.125, 0.5, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x3F80_0000, 0x3F80_0000, 0x3EC0_0000, 0x3E00_0000, 0, 0, 0, 0, 0, 0],
        }, // the flags ask for the first and third barycentrics
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0000),
            ray: ray64(0, 100.0, [0.125, 0.5, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x3F80_0000, 0x3F80_0000, 0x3EC0_0000, 0x3EC0_0000, 0, 0, 0, 0, 0, 0],
        }, // the flags ask for the first barycentric twice
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0004),
            ray: ray64(0, 100.0, [0.125, 0.5, 0.0], [0.0, 0.0, 1.0]),
            expected: [0x3F80_0000, 0x3F80_0000, 0x3EC0_0000, 0x3F00_0000, 0, 0, 0, 0, 0, 0],
        }, // the flags ask for the first and second barycentrics
        VimageCase {
            node: tri_pair_node(SQUARE, [0x11, 0x22], 0x0908),
            ray: ray64(0, 100.0, [0.25, 0.25, 2.0], [0.0, 0.0, 1.0]),
            expected: [0x7F80_0000, 0x3F80_0000, 0x3F00_0000, 0x3E80_0000, 0, 0, 0, 0, 0, 0],
        }, // the triangle is behind the ray, which is a miss with the
           // barycentrics of the crossing kept
    ]);
}

#[test]
fn image_bvh8_intersect_ray_box() {
    // An eight-wide box node returns eight child pointers, each of them the
    // child's address in the BVH -- counted from the node's own base and moved
    // on by the ranges of the children before it -- with its node type in the
    // low four bits.
    bvh8(&[
        VimageCase {
            node: box8(1, |_| 0xFF),
            ray: ray8(5, 100.0, 0xFF, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
            expected: [
                0x0000_0015, 0x0000_0025, 0x0000_0035, 0x0000_0045, 0x0000_0055, 0x0000_0065,
                0x0000_0075, 0x0000_0085, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }, // all eight children hit, in intersection order
        VimageCase {
            node: box8(1, |_| 0xFF),
            ray: ray8(5, 4.0, 0xFF, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
            expected: [
                0x0000_0015, 0x0000_0025, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
                0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }, // the ray extent clips all but the two nearest children
        VimageCase {
            node: box8(1, |_| 0xFF),
            ray: ray8(5, 100.0, 0x00, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
            expected: [
                0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
                0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }, // the ray's instance mask culls every child
        VimageCase {
            node: box8(2, |i| 1 << i),
            ray: ray8(5, 100.0, 0x55, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
            expected: [
                0x0000_0015, 0x0000_0055, 0x0000_0095, 0x0000_00D5, 0xFFFF_FFFF, 0xFFFF_FFFF,
                0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }, // a mask that keeps every other child
        VimageCase {
            node: box8(2, |i| 1 << i),
            ray: ray8(5, 100.0, 0xFF, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
            expected: [
                0x0000_0015, 0x0000_0035, 0x0000_0055, 0x0000_0075, 0x0000_0095, 0x0000_00B5,
                0x0000_00D5, 0x0000_00F5, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }, // ranges of two, which move each child's address on by two
    ]);
}

#[test]
fn image_bvh8_intersect_ray_triangle() {
    // A triangle packet returns, for each of the pair's two triangles, the
    // intersection time and two barycentrics -- already divided, unlike the
    // fraction image_bvh64_intersect_ray returns -- and the triangle's
    // identity: its primitive index doubled, with the low bit saying which way
    // round the triangle faced. The last two dwords say whether the pair ends
    // the node and whether it ends the range.
    bvh8(&[
        VimageCase {
            node: packet(),
            ray: ray8(0, 100.0, 0xFF, [0.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [
                0x3F80_0000, 0x3E80_0000, 0x3E80_0000, 0x0000_006F, 0x7F80_0000, 0xBFE0_0000,
                0x3E80_0000, 0x0000_0043, 0x0000_0000, 0x0000_0000,
            ],
        }, // the first pair: its first triangle is hit and its second missed
        VimageCase {
            node: packet(),
            ray: ray8(0, 100.0, 0xFF, [0.25, 0.25, 2.0], [0.0, 0.0, -1.0]),
            expected: [
                0x3F80_0000, 0x3E80_0000, 0x3E80_0000, 0x0000_006E, 0x7F80_0000, 0xBFE0_0000,
                0x3E80_0000, 0x0000_0042, 0x0000_0000, 0x0000_0000,
            ],
        }, // the same pair from the other side, which clears the facing bit
        VimageCase {
            node: packet(),
            ray: ray8(0, 100.0, 0xFF, [2.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [
                0x7F80_0000, 0x4010_0000, 0x3E80_0000, 0x0000_006F, 0x3F80_0000, 0x3E80_0000,
                0x3E80_0000, 0x0000_0043, 0x0000_0000, 0x0000_0000,
            ],
        }, // the pair's second triangle is the one hit
        VimageCase {
            node: packet(),
            ray: ray8(1, 100.0, 0xFF, [2.25, 0.25, 0.0], [0.0, 0.0, 1.0]),
            expected: [
                0x3F80_0000, 0x3E80_0000, 0x3E80_0000, 0x0000_0051, 0x7F80_0000, 0xFFC0_0000,
                0xFFC0_0000, 0x0000_0052, 0x0000_0003, 0x0000_0003,
            ],
        }, // the last pair, which ends both the node and the range, and whose
           // second triangle is degenerate
        VimageCase {
            node: packet(),
            ray: ray8(0, 100.0, 0xFF, [9.0, 9.0, 0.0], [0.0, 0.0, 1.0]),
            expected: [
                0x7F80_0000, 0x4110_0000, 0x4110_0000, 0x0000_006F, 0x7F80_0000, 0x40E0_0000,
                0x4110_0000, 0x0000_0043, 0x0000_0000, 0x0000_0000,
            ],
        }, // neither triangle hit, with the barycentrics of both kept
    ]);
}

#[test]
fn image_bvh64_intersect_ray_resource_base() {
    // The address of a node is the resource's base address plus the node
    // pointer, so the same node is reached however the two are split. The node
    // sits 512 bytes into the buffer, and the resource is based at its start,
    // 256 bytes in, and at the node itself.
    let mut node = vec![0u32; 128];
    node.extend(box4_types([5, 5, 5, 5]));
    let mut ray = ray64(5, 100.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);
    ray[0] = 512;
    for origin in [0, 1, 2] {
        bvh64_with(
            BVH64_RSRC,
            origin,
            &[VimageCase {
                node: node.clone(),
                ray,
                expected: [
                    0x0000_0015, 0x0000_0025, 0x0000_0035, 0x0000_0045, 0, 0, 0, 0, 0, 0,
                ],
            }],
        );
    }
}

#[test]
fn image_bvh64_intersect_ray_sorted_triangles_first() {
    // With the resource asking for it, the children that point at triangle
    // nodes come before the ones that point at boxes, each group in the order
    // the ray reaches them. Table 65 calls types 0 to 3 the triangle ones; the
    // part puts type 1 ahead of the other three as well.
    let cases = |types: [u32; 4], expected: [u32; 4]| VimageCase {
        node: box4_types(types),
        ray: ray64(5, 100.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        expected: [expected[0], expected[1], expected[2], expected[3], 0, 0, 0, 0, 0, 0],
    };

    // Without it, the children come back in the order the ray reaches them
    // whatever they point at.
    bvh64(&[
        cases([5, 0, 5, 0], [0x0000_0015, 0x0000_0020, 0x0000_0035, 0x0000_0040]),
        cases([1, 0, 2, 5], [0x0000_0011, 0x0000_0020, 0x0000_0032, 0x0000_0045]),
        cases([1, 1, 5, 0], [0x0000_0011, 0x0000_0021, 0x0000_0035, 0x0000_0040]),
    ]);

    bvh64_with(
        BVH64_RSRC_TRIANGLES_FIRST,
        0,
        &[
            cases([5, 0, 5, 0], [0x0000_0020, 0x0000_0040, 0x0000_0015, 0x0000_0035]),
            cases([1, 0, 2, 5], [0x0000_0011, 0x0000_0020, 0x0000_0032, 0x0000_0045]),
            cases([1, 1, 5, 0], [0x0000_0011, 0x0000_0021, 0x0000_0040, 0x0000_0035]),
        ],
    );
}

#[test]
fn image_bvh8_intersect_ray_sorted_triangles_first() {
    // The eight-wide instruction sorts its triangle children first too -- the
    // part answers a node with nothing at all unless the resource asks for it
    // -- and there it keeps them in the order the ray reaches them, whichever
    // of the triangle types they name.
    let children: Vec<Box8Child> = [5, 0, 5, 1, 5, 2, 5, 3]
        .iter()
        .enumerate()
        .map(|(i, &node_type)| Box8Child {
            min: [2 * i as u32 + 1, 0, 0],
            max: [2 * i as u32 + 2, 100, 100],
            node_type,
            range: 1,
            mask: 0xFF,
        })
        .collect();
    bvh8(&[VimageCase {
        node: box8_node(0x10, 0x20, [0.0, 0.0, 0.0], [139, 139, 139], &children),
        ray: ray8(5, 100.0, 0xFF, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]),
        expected: [
            0x0000_0020, 0x0000_0031, 0x0000_0042, 0x0000_0053, 0x0000_0015, 0x0000_0025,
            0x0000_0035, 0x0000_0045, 0xFFFF_FFFF, 0xFFFF_FFFF,
        ],
    }]);
}

#[test]
fn image_bvh64_intersect_ray_unsorted() {
    // With box sorting off the children come back in the order the node holds
    // them, and a child the ray never reached keeps its place rather than
    // going last. The boxes here run from the farthest to the nearest, so the
    // two orders cannot be confused.
    let node = |miss: &[usize]| {
        let mut boxes = [[0.0f32; 6]; 4];
        for (k, b) in boxes.iter_mut().enumerate() {
            let near = 2.0 * (3 - k) as f32 + 1.0;
            *b = if miss.contains(&k) {
                // behind the ray, which is a miss
                [-2.0 * k as f32 - 2.0, -1.0, -1.0, -2.0 * k as f32 - 1.0, 1.0, 1.0]
            } else {
                [near, -1.0, -1.0, near + 1.0, 1.0, 1.0]
            };
        }
        box4_node([0x15, 0x25, 0x35, 0x45], boxes)
    };
    let ray = ray64(5, 100.0, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]);

    bvh64_with(
        BVH64_RSRC_UNSORTED,
        0,
        &[
            VimageCase {
                node: node(&[]),
                ray,
                expected: [0x0000_0015, 0x0000_0025, 0x0000_0035, 0x0000_0045, 0, 0, 0, 0, 0, 0],
            }, // every child hit
            VimageCase {
                node: node(&[1]),
                ray,
                expected: [0x0000_0015, 0xFFFF_FFFF, 0x0000_0035, 0x0000_0045, 0, 0, 0, 0, 0, 0],
            }, // the second child missed, and stays where it is
            VimageCase {
                node: node(&[0, 2]),
                ray,
                expected: [0xFFFF_FFFF, 0x0000_0025, 0xFFFF_FFFF, 0x0000_0045, 0, 0, 0, 0, 0, 0],
            }, // two missed
        ],
    );

    // The same nodes with sorting on, which is the order the other tests use.
    bvh64(&[
        VimageCase {
            node: node(&[]),
            ray,
            expected: [0x0000_0045, 0x0000_0035, 0x0000_0025, 0x0000_0015, 0, 0, 0, 0, 0, 0],
        },
        VimageCase {
            node: node(&[1]),
            ray,
            expected: [0x0000_0045, 0x0000_0035, 0x0000_0015, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        },
        VimageCase {
            node: node(&[0, 2]),
            ray,
            expected: [0x0000_0045, 0x0000_0025, 0xFFFF_FFFF, 0xFFFF_FFFF, 0, 0, 0, 0, 0, 0],
        },
    ]);
}

#[test]
fn image_bvh8_intersect_ray_unsorted() {
    // The eight-wide instruction leaves its children alone the same way. Its
    // boxes run from the farthest to the nearest too.
    let children: Vec<Box8Child> = (0..8)
        .map(|i| Box8Child {
            min: [2 * (7 - i) + 1, 0, 0],
            max: [2 * (7 - i) + 2, 100, 100],
            node_type: 5,
            range: 1,
            mask: 0xFF,
        })
        .collect();
    let node = || box8_node(0x10, 0x20, [0.0, 0.0, 0.0], [139, 139, 139], &children);
    let ray = ray8(5, 100.0, 0xFF, [0.0, 5.0, 5.0], [1.0, 0.0, 0.0]);

    bvh8_with(
        BVH8_RSRC_UNSORTED,
        &[VimageCase {
            node: node(),
            ray,
            expected: [
                0x0000_0015, 0x0000_0025, 0x0000_0035, 0x0000_0045, 0x0000_0055, 0x0000_0065,
                0x0000_0075, 0x0000_0085, 0xFFFF_FFFF, 0xFFFF_FFFF,
            ],
        }],
    );
    bvh8(&[VimageCase {
        node: node(),
        ray,
        expected: [
            0x0000_0085, 0x0000_0075, 0x0000_0065, 0x0000_0055, 0x0000_0045, 0x0000_0035,
            0x0000_0025, 0x0000_0015, 0xFFFF_FFFF, 0xFFFF_FFFF,
        ],
    }]);
}
