use itertools::Itertools;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: [f32; 3],
    pub max: [f32; 3],
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
    if denom == 0.0 {
        let result0 = f32::INFINITY;
        let result1 = 0.0;
        let result2 = 0.0;
        let result3 = 0.0;

        return (result0, result1, result2, result3);
    }
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
    let barycentrics = [b_x, b_y, b_z];

    let result0 = if (denom > 0.0)
        && (b_y < 0.0 || b_y > denom || b_z < 0.0 || (b_y + b_z) > denom || (t < 0.0))
    {
        f32::INFINITY
    } else if (denom < 0.0)
        && (b_y > 0.0 || b_y < denom || b_z > 0.0 || (b_y + b_z) < denom || (t > 0.0))
    {
        f32::INFINITY
    } else {
        t
    };

    let result1 = denom;
    let result2 = barycentrics[((flags >> 0) & 3) as usize];
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
    let node_type = (node_addr & 0x7) as u8;
    match node_type {
        5 => {
            let node_ptr = (node_addr & !0x7u64) << 3;
            let node = unsafe { *(node_ptr as *const Box4Node) };

            let mut s0 = intersect(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                &node.aabb[0],
                ray_extent,
            );
            let mut s1 = intersect(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                &node.aabb[1],
                ray_extent,
            );
            let mut s2 = intersect(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                &node.aabb[2],
                ray_extent,
            );
            let mut s3 = intersect(
                [ray_origin_x, ray_origin_y, ray_origin_z],
                [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                &node.aabb[3],
                ray_extent,
            );

            let mut result0 = if s0.0 <= s0.1 {
                node.child_index[0]
            } else {
                0xFFFF_FFFF
            };
            let mut result1 = if s1.0 <= s1.1 {
                node.child_index[1]
            } else {
                0xFFFF_FFFF
            };
            let mut result2 = if s2.0 <= s2.1 {
                node.child_index[2]
            } else {
                0xFFFF_FFFF
            };
            let mut result3 = if s3.0 <= s3.1 {
                node.child_index[3]
            } else {
                0xFFFF_FFFF
            };

            let sort = |child_index_a: &mut u32,
                        child_index_b: &mut u32,
                        dist_a: &mut f32,
                        dist_b: &mut f32| {
                if (*child_index_b != 0xFFFF_FFFF && dist_b < dist_a)
                    || *child_index_a == 0xFFFF_FFFF
                {
                    let t0 = *dist_a;
                    let t1 = *child_index_a;
                    *child_index_a = *child_index_b;
                    *dist_a = *dist_b;
                    *child_index_b = t1;
                    *dist_b = t0;
                }
            };

            sort(&mut result0, &mut result2, &mut s0.0, &mut s2.0);
            sort(&mut result1, &mut result3, &mut s1.0, &mut s3.0);
            sort(&mut result0, &mut result1, &mut s0.0, &mut s1.0);
            sort(&mut result2, &mut result3, &mut s2.0, &mut s3.0);
            sort(&mut result1, &mut result2, &mut s1.0, &mut s2.0);

            unsafe {
                *result0_ptr = result0;
                *result1_ptr = result1;
                *result2_ptr = result2;
                *result3_ptr = result3;
            }
        }
        0 | 1 => {
            let node_ptr = (node_addr & !(0x7u64)) << 3;
            let node = unsafe { *(node_ptr as *const TrianglePairNode) };
            let tri = if node_type & 1 == 0 {
                [node.tri_pair.v0, node.tri_pair.v1, node.tri_pair.v2]
            } else {
                [node.tri_pair.v3, node.tri_pair.v2, node.tri_pair.v1]
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
) -> (f32, f32, f32) {
    let e1 = [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]];
    let e2 = [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]];
    let s1 = cross(ray_direction, e2);
    let denom = dot(s1, e1);
    if denom == 0.0 {
        let result0 = f32::INFINITY;
        let result1 = 0.0;
        let result2 = 0.0;

        return (result0, result1, result2);
    }
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

    let t = if b_y < 0.0 || b_y > 1.0 || b_z < 0.0 || (b_y + b_z) > 1.0 {
        f32::INFINITY
    } else {
        t
    };

    (t, b_y, b_z)
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
    node_base: u64,
    ray_extent: f32,
    ray_origin_x: f32,
    ray_origin_y: f32,
    ray_origin_z: f32,
    ray_dir_x: f32,
    ray_dir_y: f32,
    ray_dir_z: f32,
    node_index: u32,
) {
    let node_ptr = (node_base + (node_index & !0xF) as u64) << 3;
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

            let prim0 = node.get_prim_index(tri_pair_index as u32, 0);
            let prim1 = node.get_prim_index(tri_pair_index as u32, 1);

            let node_end = (tri_pair_index as u32 + 1) == node.get_triangle_pair_count();
            let range_end = node.is_range_end(tri_pair_index as u32);

            unsafe {
                *result0_ptr = f32::to_bits(result0.0);
                *result1_ptr = f32::to_bits(result0.1);
                *result2_ptr = f32::to_bits(result0.2);
                *result3_ptr = prim0;
                *result4_ptr = f32::to_bits(result1.0);
                *result5_ptr = f32::to_bits(result1.1);
                *result6_ptr = f32::to_bits(result1.2);
                *result7_ptr = prim1;
                *result8_ptr = ((range_end as u32) << 1) | (node_end as u32);
                *result9_ptr = 0;
            }
        }
        5 => {
            let node = unsafe { *(node_ptr as *const Box8Node) };
            let ray_inv_dir_x = 1.0 / ray_dir_x;
            let ray_inv_dir_y = 1.0 / ray_dir_y;
            let ray_inv_dir_z = 1.0 / ray_dir_z;

            let child_count = node.get_child_count();

            let boxes = (0..child_count)
                .map(|i| node.get_child_box(i as usize))
                .collect::<Vec<Aabb>>();

            let s = boxes
                .iter()
                .map(|aabb| {
                    intersect(
                        [ray_origin_x, ray_origin_y, ray_origin_z],
                        [ray_inv_dir_x, ray_inv_dir_y, ray_inv_dir_z],
                        aabb,
                        ray_extent,
                    )
                })
                .collect::<Vec<(f32, f32)>>();

            let results = (0..8)
                .map(|i| {
                    if i < 8 && s[i as usize].0 <= s[i as usize].1 {
                        node.get_child_index(i as usize)
                    } else {
                        0xFFFF_FFFF
                    }
                })
                .collect::<Vec<u32>>();

            let results = results
                .into_iter()
                .zip(s.into_iter())
                .sorted_by(|&(_idx_a, (dist_a, _)), &(_idx_b, (dist_b, _))| {
                    if (_idx_b != 0xFFFF_FFFF && dist_b < dist_a) || _idx_a == 0xFFFF_FFFF {
                        std::cmp::Ordering::Greater
                    } else {
                        std::cmp::Ordering::Less
                    }
                })
                .map(|(idx, _)| idx)
                .collect::<Vec<u32>>();

            unsafe {
                *result0_ptr = results[0];
                *result1_ptr = results[1];
                *result2_ptr = results[2];
                *result3_ptr = results[3];
                *result4_ptr = results[4];
                *result5_ptr = results[5];
                *result6_ptr = results[6];
                *result7_ptr = results[7];
                *result8_ptr = 0;
                *result9_ptr = 0;
            }
        }
        _ => {
            panic!("Unsupported node type: {}", node_type);
        }
    }
}
