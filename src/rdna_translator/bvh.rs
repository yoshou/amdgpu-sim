use crate::buffer::get_bits_u32;
use itertools::Itertools;

/// Where a coordinate lands in a row or a column of `size` texels, or None
/// when it lands on the border instead. The modes are the ones Table 61 names.
fn clamp_texel(coord: i32, size: i32, mode: u32) -> Option<i32> {
    // Mirroring once folds the coordinates below the image back over it.
    let mirror_once = |coord: i32| if coord < 0 { -1 - coord } else { coord };
    match mode {
        0 => Some(coord.rem_euclid(size)),
        1 => {
            let folded = coord.rem_euclid(2 * size);
            Some(if folded < size {
                folded
            } else {
                2 * size - 1 - folded
            })
        }
        2 | 4 => Some(coord.clamp(0, size - 1)),
        3 | 5 => Some(mirror_once(coord).clamp(0, size - 1)),
        6 => (0..size).contains(&coord).then_some(coord),
        7 => {
            let mirrored = mirror_once(coord);
            (0..size).contains(&mirrored).then_some(mirrored)
        }
        mode => unimplemented!("image clamp mode {}", mode),
    }
}

/// What a texel of `format` holds. Every format here has a single channel of
/// eight bits, which the part gives to whichever channel a selector names.

fn texel_value(format: u32, raw: u8) -> u32 {
    match format {
        1 => f32::to_bits(raw as f32 / 255.0),
        2 => f32::to_bits((raw as i8).max(-127) as f32 / 127.0),
        5 => raw as u32,
        6 => ((raw as i8) as i32) as u32,
        format => unimplemented!("image data format {}", format),
    }
}

/// What the selector SEL_1 and an opaque-white border stand for: one, as the
/// format counts it.
fn format_one(format: u32) -> u32 {
    match format {
        1 | 2 => f32::to_bits(1.0),
        _ => 1,
    }
}

/// Runtime helper for IMAGE_SAMPLE_LZ, called once per component the DMASK
/// asks for. Mirrors the interpreter's `image_sample_lz` (see
/// rdna_processor.rs); resolved by the JIT via the process symbol table like
/// the `image_bvh*_intersect_ray` helpers.
#[unsafe(no_mangle)]
pub extern "C" fn image_sample_lz(
    r0: u32,
    r1: u32,
    r2: u32,
    r3: u32,
    r4: u32,
    r5: u32,
    r6: u32,
    r7: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
    component: u32,
    unrm: u32,
    u: f32,
    v: f32,
) -> u32 {
    let rsrc = [r0, r1, r2, r3, r4, r5, r6, r7];
    let samp = [s0, s1, s2, s3];

    let format = get_bits_u32(&rsrc, 49, 8);
    let width = get_bits_u32(&rsrc, 62, 16) + 1;
    let height = get_bits_u32(&rsrc, 78, 16) + 1;
    let base_addr = (((rsrc[1] as u64) << 40) | ((rsrc[0] as u64) << 8)) & ((1u64 << 48) - 1);
    // The resource can give a row pitch of its own, and the part rounds a row
    // up to 128 bytes whatever it says. The formats read here are a byte a
    // texel, so a row of texels is a row of bytes.
    let pitch = get_bits_u32(&rsrc, 128, 14) | (get_bits_u32(&rsrc, 142, 2) << 14);
    let row = if pitch != 0 { pitch + 1 } else { width }.max(128) as u64;

    if get_bits_u32(&samp, 84, 2) != 0 {
        unimplemented!("IMAGE_SAMPLE_LZ with a filter other than point");
    }

    // What the selector this component names asks for: a constant needs
    // neither an address nor a texel.
    match get_bits_u32(&rsrc, 96 + component as usize * 3, 3) {
        0 => return 0,
        1 => return format_one(format),
        4..=7 => {}
        select => unimplemented!("image destination select {}", select),
    }

    // The instruction can force the address to be unnormalized, and so can the
    // sampler; otherwise it spans the image.
    let unnormalized = unrm != 0 || get_bits_u32(&samp, 15, 1) != 0;
    let texel = |coord: f32, size: u32| {
        if unnormalized {
            coord
        } else {
            coord * size as f32
        }
        .floor() as i32
    };
    // Unnormalized coordinates cannot repeat the image, so the two modes that
    // repeat it fold it over once instead.
    let mode = |axis: usize| match get_bits_u32(&samp, axis * 3, 3) {
        0 if unnormalized => 2,
        1 if unnormalized => 3,
        mode => mode,
    };
    let x = clamp_texel(texel(u, width), width as i32, mode(0));
    let y = clamp_texel(texel(v, height), height as i32, mode(1));

    match (x, y) {
        (Some(x), Some(y)) => {
            let ptr = (base_addr + y as u64 * row + x as u64) as *const u8;
            texel_value(format, unsafe { std::ptr::read_unaligned(ptr) })
        }
        // Off the image, the sampler's border colour stands in for the texel.
        _ => match get_bits_u32(&samp, 126, 2) {
            0 | 1 => 0,
            2 => format_one(format),
            _ => unimplemented!("IMAGE_SAMPLE_LZ with a border colour of its own"),
        },
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// Where a child pointer sorts when the resource asks for triangle nodes to
/// come first. Table 65 puts them before the box nodes -- types 0 to 3 for the
/// four-wide instructions -- and the part puts type 1 before the other three
/// as well.
fn box4_child_rank(node_type: u32) -> u32 {
    match node_type {
        1 => 0,
        0 | 2 | 3 => 1,
        _ => 2,
    }
}

/// The same for the eight-wide instruction, whose triangle nodes are the
/// packet types and which keeps them in the order the ray reaches them.
fn box8_child_rank(node_type: u32) -> u32 {
    match node_type {
        0..=3 | 8..=11 => 0,
        _ => 1,
    }
}

/// Whether the child `b` belongs before the child `a` when the resource sorts
/// its boxes and nothing else: one the ray never reached goes last, and the
/// rest go by the time the ray enters them.
fn closer_than(b: (u32, u32, f32), a: (u32, u32, f32)) -> bool {
    (b.0 != 0xFFFF_FFFF && b.2 < a.2) || a.0 == 0xFFFF_FFFF
}

/// The network the part sorts its four children with.
const BOX4_NETWORK: [(usize, usize); 5] = [(0, 2), (1, 3), (0, 1), (2, 3), (1, 2)];

/// Whether the child `b` belongs before the child `a`: one the ray never
/// reached goes last, and the rest go by rank and then by the time the ray
/// enters them. Each child is its pointer, its rank and that time.
fn sorts_before(b: (u32, u32, f32), a: (u32, u32, f32)) -> bool {
    (b.0 != 0xFFFF_FFFF && (b.1 < a.1 || (b.1 == a.1 && b.2 < a.2))) || a.0 == 0xFFFF_FFFF
}

/// The base of the BVH the resource names, which the node pointers count from.
/// Table 65 gives it in 256-byte units.
fn bvh_base_addr(r0: u32, r1: u32) -> u64 {
    ((((r1 as u64) & 0xFF) << 32) | (r0 as u64)) << 8
}

/// Whether the resource asks for triangle nodes to be sorted first.
fn bvh_sorts_triangles_first(r1: u32) -> bool {
    (r1 >> 20) & 1 != 0
}

/// Whether it asks for the children to be sorted at all. Without it they come
/// back in the order the node holds them.
fn bvh_sorts_boxes(r1: u32) -> bool {
    (r1 >> 31) & 1 != 0
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Box4Node {
    pub child_index: [u32; 4],
    pub aabb: [Aabb; 4],
    pub parent_addr: u32,
    pub update_counter: u32,
    pub child_count: u32,
}

/// Ordered min/max: `minps`/`maxps` semantics, one instruction. Equal to
/// `f32::min`/`f32::max` (IEEE minNum/maxNum) whenever neither operand is NaN;
/// see [`intersect4`] for why that precondition is testable in bulk.
#[inline(always)]
fn rmin(a: f32, b: f32) -> f32 {
    if a < b {
        a
    } else {
        b
    }
}

#[inline(always)]
fn rmax(a: f32, b: f32) -> f32 {
    if a > b {
        a
    } else {
        b
    }
}

/// Slab test for all four children of a [`Box4Node`] at once, SoA over the
/// children. Bit-identical to four [`intersect`] calls; returns `(t0, t1)` per
/// child.
///
/// `f32::min`/`f32::max` are IEEE minNum/maxNum, which x86 implements as a
/// min/max plus a three-instruction NaN blend — twelve of them per node here.
/// A slab value is NaN only when an axis multiplies `0 * inf` (a ray parallel
/// to a slab whose origin lies exactly on one of the box planes), so testing
/// all 24 of them for NaN once (six vector compares) lets the common case use
/// the raw ordered min/max, with the minNum form kept for the rare case.
#[inline]
fn intersect4(
    ray_origin: [f32; 3],
    inv_direction: [f32; 3],
    aabb: &[Aabb; 4],
    max_t: f32,
) -> ([f32; 4], [f32; 4]) {
    let mut f = [[0f32; 4]; 3];
    let mut n = [[0f32; 4]; 3];
    for axis in 0..3 {
        for child in 0..4 {
            f[axis][child] = (aabb[child].max[axis] - ray_origin[axis]) * inv_direction[axis];
            n[axis][child] = (aabb[child].min[axis] - ray_origin[axis]) * inv_direction[axis];
        }
    }

    let mut has_nan = false;
    for axis in 0..3 {
        for child in 0..4 {
            has_nan |= f[axis][child].is_nan() | n[axis][child].is_nan();
        }
    }

    let mut t0 = [0f32; 4];
    let mut t1 = [0f32; 4];
    if has_nan {
        for c in 0..4 {
            let hi = [
                f[0][c].max(n[0][c]),
                f[1][c].max(n[1][c]),
                f[2][c].max(n[2][c]),
            ];
            let lo = [
                f[0][c].min(n[0][c]),
                f[1][c].min(n[1][c]),
                f[2][c].min(n[2][c]),
            ];
            t1[c] = hi[0].min(hi[1].min(hi[2].min(max_t)));
            t0[c] = lo[0].max(lo[1].max(lo[2].max(0.0)));
        }
    } else {
        for c in 0..4 {
            let hi = [
                rmax(f[0][c], n[0][c]),
                rmax(f[1][c], n[1][c]),
                rmax(f[2][c], n[2][c]),
            ];
            let lo = [
                rmin(f[0][c], n[0][c]),
                rmin(f[1][c], n[1][c]),
                rmin(f[2][c], n[2][c]),
            ];
            t1[c] = rmin(hi[0], rmin(hi[1], rmin(hi[2], max_t)));
            t0[c] = rmax(lo[0], rmax(lo[1], rmax(lo[2], 0.0)));
        }
    }
    (t0, t1)
}

fn intersect(ray_origin: [f32; 3], inv_direction: [f32; 3], aabb: &Aabb, max_t: f32) -> (f32, f32) {
    let f = [
        (aabb.max[0] - ray_origin[0]) * inv_direction[0],
        (aabb.max[1] - ray_origin[1]) * inv_direction[1],
        (aabb.max[2] - ray_origin[2]) * inv_direction[2],
    ];
    let n = [
        (aabb.min[0] - ray_origin[0]) * inv_direction[0],
        (aabb.min[1] - ray_origin[1]) * inv_direction[1],
        (aabb.min[2] - ray_origin[2]) * inv_direction[2],
    ];
    let tmax = [f[0].max(n[0]), f[1].max(n[1]), f[2].max(n[2])];
    let tmin = [f[0].min(n[0]), f[1].min(n[1]), f[2].min(n[2])];
    let t1 = tmax[0].min(tmax[1].min(tmax[2].min(max_t)));
    let t0 = tmin[0].max(tmin[1].max(tmin[2].max(0.0)));
    (t0, t1)
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn intersect_triangle_frac(
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    flags: u32,
) -> (f32, f32, f32, f32) {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let s1 = cross(ray_direction, e2);
    let denom = dot(s1, e1);
    let d = [
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2],
    ];
    let b_y = dot(d, s1);
    let s2 = cross(d, e1);
    let b_z = dot(ray_direction, s2);
    let t: f32 = dot(e2, s2);
    let b_x = denom - b_y - b_z;
    // The flags name a barycentric in two bits each; the fourth encoding is
    // reserved, and the part answers it with the first barycentric.
    let barycentrics = [b_x, b_y, b_z, b_x];

    let hit = if denom > 0.0 {
        !(b_y < 0.0 || b_y > denom || b_z < 0.0 || (b_y + b_z) > denom || (t < 0.0))
    } else if denom < 0.0 {
        !(b_y > 0.0 || b_y < denom || b_z > 0.0 || (b_y + b_z) < denom || (t > 0.0))
    } else {
        // A ray running in the plane of the triangle meets nothing.
        false
    };

    // A miss is an infinite numerator, signed with the denominator so that the
    // quotient is positive infinity whichever way round the triangle faces.
    let result0 = if hit {
        t
    } else {
        f32::INFINITY.copysign(denom)
    };
    let result1 = denom;
    let result2 = barycentrics[(flags & 3) as usize];
    let result3 = barycentrics[((flags >> 2) & 3) as usize];

    (result0, result1, result2, result3)
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct TrianglePair {
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
    v3: [f32; 3],
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
struct TrianglePairNode {
    pub tri_pair: TrianglePair,
    pub padding: u32,
    pub prim_index: [u32; 2],
    pub flags: u32,
}

#[unsafe(no_mangle)]
pub extern "C" fn image_bvh64_intersect_ray(
    result0_ptr: *mut u32,
    result1_ptr: *mut u32,
    result2_ptr: *mut u32,
    result3_ptr: *mut u32,
    r0: u32,
    r1: u32,
    node_addr: u64,
    ray_extent: f32,
    ray_origin_x: f32,
    ray_origin_y: f32,
    ray_origin_z: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_dir_z: f32,
    ray_inv_dir_x: f32,
    ray_inv_dir_y: f32,
    ray_inv_dir_z: f32,
) {
    let base_addr = bvh_base_addr(r0, r1);
    let sort_triangles_first = bvh_sorts_triangles_first(r1);
    let box_sort = bvh_sorts_boxes(r1);
    let node_type = (node_addr & 0x7) as u8;
    match node_type {
        5 => {
            let node_ptr = base_addr + ((node_addr & !0x7u64) << 3);
            let node = unsafe { *(node_ptr as *const Box4Node) };

            let (t0, t1) = intersect4(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                &node.aabb,
                ray_extent,
            );

            let mut children = [(0u32, 0u32, 0.0f32); 4];
            for (i, child) in children.iter_mut().enumerate() {
                let index = node.child_index[i];
                let rank = if sort_triangles_first {
                    box4_child_rank(index & 7)
                } else {
                    0
                };
                *child = (
                    if t0[i] <= t1[i] { index } else { 0xFFFF_FFFF },
                    rank,
                    t0[i],
                );
            }

            // The children are sorted only if the resource asks for it;
            // otherwise they come back in the order the node holds them.
            // Ranking them is work the common case does not need, so the two
            // orders have a pass each.
            if box_sort && sort_triangles_first {
                for (a, b) in BOX4_NETWORK {
                    if sorts_before(children[b], children[a]) {
                        children.swap(a, b);
                    }
                }
            } else if box_sort {
                for (a, b) in BOX4_NETWORK {
                    if closer_than(children[b], children[a]) {
                        children.swap(a, b);
                    }
                }
            }

            unsafe {
                *result0_ptr = children[0].0;
                *result1_ptr = children[1].0;
                *result2_ptr = children[2].0;
                *result3_ptr = children[3].0;
            }
        }
        0 | 1 => {
            let node_ptr = base_addr + ((node_addr & !(0x7u64)) << 3);
            let node = unsafe { *(node_ptr as *const TrianglePairNode) };
            let tri = if node_type & 1 == 0 {
                [node.tri_pair.v0, node.tri_pair.v1, node.tri_pair.v2]
            } else {
                // The pair's second triangle, wound the way the part takes
                // it: the same three vertices as (v3, v2, v1), rotated so that
                // the barycentrics come back in the order the flags name.
                [node.tri_pair.v1, node.tri_pair.v3, node.tri_pair.v2]
            };
            let result = intersect_triangle_frac(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_dir_x, ray_dir_y, ray_dir_z],
                tri[0],
                tri[1],
                tri[2],
                node.flags >> ((node_type & 1) * 8),
            );

            unsafe {
                *result0_ptr = f32::to_bits(result.0);
                *result1_ptr = f32::to_bits(result.1);
                *result2_ptr = f32::to_bits(result.2);
                *result3_ptr = f32::to_bits(result.3);
            }
        }
        _ => {
            panic!("Unsupported node type: {}", node_type);
        }
    };
}

pub const BVH_RAY_PACKET_LANES: usize = 16;

#[repr(C, align(64))]
pub struct BvhRayPacket {
    pub node_addr: [u64; BVH_RAY_PACKET_LANES],
    pub ray_extent: [f32; BVH_RAY_PACKET_LANES],
    pub ray_origin_x: [f32; BVH_RAY_PACKET_LANES],
    pub ray_origin_y: [f32; BVH_RAY_PACKET_LANES],
    pub ray_origin_z: [f32; BVH_RAY_PACKET_LANES],
    pub ray_dir_x: [f32; BVH_RAY_PACKET_LANES],
    pub ray_dir_y: [f32; BVH_RAY_PACKET_LANES],
    pub ray_dir_z: [f32; BVH_RAY_PACKET_LANES],
    pub ray_inv_dir_x: [f32; BVH_RAY_PACKET_LANES],
    pub ray_inv_dir_y: [f32; BVH_RAY_PACKET_LANES],
    pub ray_inv_dir_z: [f32; BVH_RAY_PACKET_LANES],
    pub result0: [u32; BVH_RAY_PACKET_LANES],
    pub result1: [u32; BVH_RAY_PACKET_LANES],
    pub result2: [u32; BVH_RAY_PACKET_LANES],
    pub result3: [u32; BVH_RAY_PACKET_LANES],
}

#[unsafe(no_mangle)]
pub extern "C" fn image_bvh64_intersect_ray_packet(
    packet: *mut BvhRayPacket,
    lane_count: u32,
    active_mask: u32,
    r0: u32,
    r1: u32,
) {
    assert!(lane_count as usize <= BVH_RAY_PACKET_LANES);
    let packet = unsafe { &mut *packet };
    for lane in 0..lane_count as usize {
        if active_mask & (1u32 << lane) == 0 {
            packet.result0[lane] = 0;
            packet.result1[lane] = 0;
            packet.result2[lane] = 0;
            packet.result3[lane] = 0;
            continue;
        }
        image_bvh64_intersect_ray(
            &mut packet.result0[lane],
            &mut packet.result1[lane],
            &mut packet.result2[lane],
            &mut packet.result3[lane],
            r0,
            r1,
            packet.node_addr[lane],
            packet.ray_extent[lane],
            packet.ray_origin_x[lane],
            packet.ray_origin_y[lane],
            packet.ray_origin_z[lane],
            packet.ray_dir_x[lane],
            packet.ray_dir_y[lane],
            packet.ray_dir_z[lane],
            packet.ray_inv_dir_x[lane],
            packet.ray_inv_dir_y[lane],
            packet.ray_inv_dir_z[lane],
        );
    }
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
pub struct Box8Node {
    data: [u32; 32],
}

impl Box8Node {
    pub fn get_box_node_base(&self) -> u32 {
        self.data[0]
    }

    pub fn get_prim_node_base(&self) -> u32 {
        self.data[1]
    }

    pub fn get_parent_addr(&self) -> u32 {
        self.data[2]
    }

    pub fn get_origin(&self) -> [f32; 3] {
        let x = f32::from_bits(self.data[3]);
        let y = f32::from_bits(self.data[4]);
        let z = f32::from_bits(self.data[5]);
        [x, y, z]
    }

    pub fn get_exponent(&self) -> [u8; 3] {
        let x = (self.data[6] & 0xFF) as u8;
        let y = ((self.data[6] >> 8) & 0xFF) as u8;
        let z = ((self.data[6] >> 16) & 0xFF) as u8;
        [x, y, z]
    }

    pub fn get_child_count(&self) -> u8 {
        ((self.data[6] >> 28) & 0x0F) as u8 + 1
    }

    pub fn get_matrix_id(&self) -> u32 {
        (self.data[7] as u32) & 0x7F
    }

    pub fn get_child_box(&self, index: usize) -> Aabb {
        let exponent = self.get_exponent();
        let origin = self.get_origin();

        let rcp_exponent = [
            f32::from_bits((254 - (exponent[0] as u32) + 12) << 23),
            f32::from_bits((254 - (exponent[1] as u32) + 12) << 23),
            f32::from_bits((254 - (exponent[2] as u32) + 12) << 23),
        ];

        let min_x = origin[0] + (self.data[8 + index * 3] & 0x00000FFF) as f32 / rcp_exponent[0];
        let min_y =
            origin[1] + ((self.data[8 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[1];
        let min_z = origin[2] + ((self.data[9 + index * 3]) & 0x00000FFF) as f32 / rcp_exponent[2];
        let max_x = origin[0]
            + if exponent[0] != 0 {
                ((self.data[9 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[0]
            } else {
                0.0
            };
        let max_y = origin[1]
            + if exponent[1] != 0 {
                (self.data[10 + index * 3] & 0x00000FFF) as f32 / rcp_exponent[1]
            } else {
                0.0
            };
        let max_z = origin[2]
            + if exponent[2] != 0 {
                ((self.data[10 + index * 3] >> 12) & 0x00000FFF) as f32 / rcp_exponent[2]
            } else {
                0.0
            };

        Aabb {
            min: [min_x, min_y, min_z],
            max: [max_x, max_y, max_z],
        }
    }

    pub fn get_child_type(&self, index: usize) -> u8 {
        ((self.data[10 + index * 3] >> 24) & 0x0F) as u8
    }

    pub fn get_child_addr(&self, index: usize) -> u32 {
        let child_type = self.get_child_type(index);
        let mut child_addr = if child_type == 5 {
            self.data[0] >> 4
        } else {
            self.data[1] >> 4
        };
        for j in 0..index {
            if (self.get_child_type(j) == 5) == (child_type == 5) {
                let node_range = (self.data[10 + j * 3] >> 28) & 0x0F;
                child_addr += node_range;
            }
        }
        child_addr
    }

    /// The instance mask that can cull this child: the ray carries one of its
    /// own, and a child whose mask has no bit in common with it is skipped.
    pub fn get_child_mask(&self, index: usize) -> u32 {
        (self.data[9 + index * 3] >> 24) & 0xFF
    }

    pub fn get_child_index(&self, index: usize) -> u32 {
        (self.get_child_addr(index) << 4) | (self.get_child_type(index) as u32)
    }
}

#[repr(C, align(64))]
#[derive(Debug, Clone, Copy)]
struct TrianglePacketNode {
    data: [u32; 32],
}

impl TrianglePacketNode {
    pub fn read_unaligned_bits(&self, position: u32, length: u32) -> u32 {
        let mut data = 0u64;
        if length != 0 {
            data = self.data[(position / 32) as usize] as u64;
            if (position + length - 1) / 32 != position / 32 {
                data |= (self.data[((position + length - 1) / 32) as usize] as u64) << 32;
            }
            data >>= position % 32;
            data &= (1 << length) - 1;
        }
        data as u32
    }

    pub fn read_vertex(&self, vertex_index: u32) -> [f32; 3] {
        let position = 52 + 96 * vertex_index;

        let x_bits = self.read_unaligned_bits(position + 0 * 32, 32);
        let y_bits = self.read_unaligned_bits(position + 1 * 32, 32);
        let z_bits = self.read_unaligned_bits(position + 2 * 32, 32);

        [
            f32::from_bits(x_bits),
            f32::from_bits(y_bits),
            f32::from_bits(z_bits),
        ]
    }

    pub fn read_descriptor(&self, pair_index: u32, triangle_index: u32) -> [u32; 4] {
        let position = 1024 - (pair_index + 1) * 29;
        let descriptor = self.read_unaligned_bits(position, 29);
        let tri_indices = if triangle_index > 0 {
            descriptor >> 3
        } else {
            descriptor >> 17
        };
        [
            tri_indices & 15,
            (tri_indices >> 4) & 15,
            (tri_indices >> 8) & 15,
            descriptor & 1,
        ]
    }

    pub fn fetch_triangle(&self, pair_index: u32, triangle_index: u32) -> [[f32; 3]; 3] {
        let tri_indices = self.read_descriptor(pair_index, triangle_index);

        let v0 = self.read_vertex(tri_indices[0]);
        let v1 = self.read_vertex(tri_indices[1]);
        let v2 = self.read_vertex(tri_indices[2]);

        [v0, v1, v2]
    }

    pub fn get_triangle_pair_count(&self) -> u32 {
        self.read_unaligned_bits(28, 3) + 1
    }

    pub fn get_index_section_midpoint(&self) -> u32 {
        self.read_unaligned_bits(32 + 10, 10)
    }

    pub fn get_prim_index_anchor_size(&self) -> u32 {
        self.read_unaligned_bits(32 + 0, 5)
    }

    pub fn get_prim_index_payload_size(&self) -> u32 {
        self.read_unaligned_bits(32 + 5, 5)
    }

    pub fn read_prim_index(&self, pair_index: u32, triangle_index: u32) -> u32 {
        let flat_tri_index = 2 * pair_index + triangle_index;

        let prim_index_payload_size = self.get_prim_index_payload_size();
        let prim_index_anchor_size = self.get_prim_index_anchor_size();
        let prim_index_anchor_pos = self.get_index_section_midpoint();

        let prim_index_anchor =
            self.read_unaligned_bits(prim_index_anchor_pos, prim_index_anchor_size);
        if flat_tri_index == 0 {
            return prim_index_anchor;
        }
        let prim_index_payload_pos = prim_index_anchor_pos
            + prim_index_anchor_size
            + (flat_tri_index - 1) * prim_index_payload_size;

        let prim_index = self.read_unaligned_bits(prim_index_payload_pos, prim_index_payload_size);
        let prim_index_mask = (1 << prim_index_payload_size) - 1;

        if prim_index_payload_size >= prim_index_anchor_size {
            prim_index
        } else {
            prim_index | (prim_index_anchor & !prim_index_mask)
        }
    }

    pub fn get_prim_index(&self, pair_index: u32, triangle_index: u32) -> u32 {
        self.read_prim_index(pair_index, triangle_index)
    }

    pub fn is_range_end(&self, pair_index: u32) -> bool {
        let descriptor = self.read_descriptor(pair_index, 0);
        descriptor[3] != 0
    }
}

fn intersect_triangle(
    ray_origin: [f32; 3],
    ray_direction: [f32; 3],
    v0: [f32; 3],
    v1: [f32; 3],
    v2: [f32; 3],
) -> (f32, f32, f32, f32) {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let s1 = cross(ray_direction, e2);
    let denom = dot(s1, e1);
    let d = [
        ray_origin[0] - v0[0],
        ray_origin[1] - v0[1],
        ray_origin[2] - v0[2],
    ];
    let inv_denom = 1.0 / denom;
    let b_y = dot(d, s1) * inv_denom;
    let s2 = cross(d, e1);
    let b_z = dot(ray_direction, s2) * inv_denom;
    let t: f32 = dot(e2, s2) * inv_denom;

    // A ray running in the plane of the triangle meets nothing, and neither
    // does one that crosses it outside its edges. The barycentrics stand
    // either way, which for a zero denominator leaves them what dividing by
    // zero makes of them.
    let t = if denom == 0.0 || b_y < 0.0 || b_y > 1.0 || b_z < 0.0 || (b_y + b_z) > 1.0 {
        f32::INFINITY
    } else {
        t
    };

    (t, b_y, b_z, denom)
}

#[unsafe(no_mangle)]
pub extern "C" fn image_bvh8_intersect_ray(
    result0_ptr: *mut u32,
    result1_ptr: *mut u32,
    result2_ptr: *mut u32,
    result3_ptr: *mut u32,
    result4_ptr: *mut u32,
    result5_ptr: *mut u32,
    result6_ptr: *mut u32,
    result7_ptr: *mut u32,
    result8_ptr: *mut u32,
    result9_ptr: *mut u32,
    r0: u32,
    r1: u32,
    node_base: u64,
    ray_extent: f32,
    instance_mask: u32,
    ray_origin_x: f32,
    ray_origin_y: f32,
    ray_origin_z: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_dir_z: f32,
    node_index: u32,
) {
    let node_ptr = bvh_base_addr(r0, r1) + ((node_base + (node_index & !0xF) as u64) << 3);
    let sort_triangles_first = bvh_sorts_triangles_first(r1);
    let box_sort = bvh_sorts_boxes(r1);
    let node_type = (node_index & 0xF) as u8;
    match node_type {
        0..3 | 8..11 => {
            let tri_pair_index = (node_type & 3) + ((node_type & 8) >> 1);
            let node = unsafe { *(node_ptr as *const TrianglePacketNode) };
            let tri0 = node.fetch_triangle(tri_pair_index as u32, 0);
            let tri1 = node.fetch_triangle(tri_pair_index as u32, 1);

            let result0 = intersect_triangle(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_dir_x, ray_dir_y, ray_dir_z],
                tri0[0],
                tri0[1],
                tri0[2],
            );
            let result1 = intersect_triangle(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_dir_x, ray_dir_y, ray_dir_z],
                tri1[0],
                tri1[1],
                tri1[2],
            );

            // A triangle is named by its primitive index doubled, with the low
            // bit saying which way round the ray met it.
            let prim0 =
                (node.get_prim_index(tri_pair_index as u32, 0) << 1) | (result0.3 < 0.0) as u32;
            let prim1 =
                (node.get_prim_index(tri_pair_index as u32, 1) << 1) | (result1.3 < 0.0) as u32;

            let node_end = (tri_pair_index as u32 + 1) == node.get_triangle_pair_count();
            let range_end = node.is_range_end(tri_pair_index as u32);
            let ends = ((range_end as u32) << 1) | (node_end as u32);

            unsafe {
                *result0_ptr = f32::to_bits(result0.0);
                *result1_ptr = f32::to_bits(result0.1);
                *result2_ptr = f32::to_bits(result0.2);
                *result3_ptr = prim0;
                *result4_ptr = f32::to_bits(result1.0);
                *result5_ptr = f32::to_bits(result1.1);
                *result6_ptr = f32::to_bits(result1.2);
                *result7_ptr = prim1;
                // The last two dwords stand for the pair's two triangles, and
                // both say where the pair ends.
                *result8_ptr = ends;
                *result9_ptr = ends;
            }
        }
        5 => {
            let node = unsafe { *(node_ptr as *const Box8Node) };
            let ray_inv_dir_x = 1.0 / ray_dir_x;
            let ray_inv_dir_y = 1.0 / ray_dir_y;
            let ray_inv_dir_z = 1.0 / ray_dir_z;

            let child_count = node.get_child_count() as usize;

            // A child the node does not have is a miss, and so is one whose
            // instance mask has nothing in common with the ray's.
            let results = (0..8)
                .map(|i| {
                    if i >= child_count {
                        return (0xFFFF_FFFF, 0, f32::INFINITY);
                    }
                    let (t0, t1) = intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        &node.get_child_box(i),
                        ray_extent,
                    );
                    let hit = t0 <= t1 && (instance_mask & node.get_child_mask(i)) != 0;
                    let index = if hit {
                        node.get_child_index(i)
                    } else {
                        0xFFFF_FFFF
                    };
                    let rank = if sort_triangles_first {
                        box8_child_rank(node.get_child_type(i) as u32)
                    } else {
                        0
                    };
                    (index, rank, t0)
                })
                .collect::<Vec<(u32, u32, f32)>>();

            // The children come back sorted only if the resource asks for
            // it; otherwise in the order the node holds them.
            let results = if box_sort {
                results
                    .into_iter()
                    .sorted_by(|&a, &b| {
                        if sorts_before(b, a) {
                            std::cmp::Ordering::Greater
                        } else {
                            std::cmp::Ordering::Less
                        }
                    })
                    .map(|(index, _, _)| index)
                    .collect::<Vec<u32>>()
            } else {
                results
                    .into_iter()
                    .map(|(index, _, _)| index)
                    .collect::<Vec<u32>>()
            };

            unsafe {
                *result0_ptr = results[0];
                *result1_ptr = results[1];
                *result2_ptr = results[2];
                *result3_ptr = results[3];
                *result4_ptr = results[4];
                *result5_ptr = results[5];
                *result6_ptr = results[6];
                *result7_ptr = results[7];
                // A box node has nothing to say in the two dwords a triangle
                // node names its triangles with.
                *result8_ptr = 0xFFFF_FFFF;
                *result9_ptr = 0xFFFF_FFFF;
            }
        }
        _ => {
            panic!("Unsupported node type: {}", node_type);
        }
    }
}
