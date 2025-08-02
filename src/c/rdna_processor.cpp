#include <immintrin.h>

#include <cstdbool>
#include <cstdint>
#include <stdexcept>
#include <tuple>

extern "C" {
void v_fma_f64(uint16_t d, uint16_t s0, uint16_t s1, uint16_t s2, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod,
               const int32_t *sgprs, int32_t *vgprs, const int32_t *literal_constant);
void v_add_f64_e64(uint16_t d, uint16_t s0, uint16_t s1, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod,
                   const int32_t *sgprs, int32_t *vgprs, const int32_t *literal_constant);
void v_mul_f64_e32(uint16_t d, uint16_t s0, uint16_t s1, const int32_t *sgprs, int32_t *vgprs,
                   const int32_t *literal_constant);
}

struct m256dx2 {
  __m256d value0;
  __m256d value1;

  m256dx2(__m256d v0, __m256d v1) : value0(v0), value1(v1) {}
  m256dx2() : value0(_mm256_setzero_pd()), value1(_mm256_setzero_pd()) {}
};

template <size_t idx>
static m256dx2 abs_neg(const m256dx2 value, uint8_t abs, uint8_t neg) {
  auto value0 = value.value0;
  auto value1 = value.value1;
  if ((abs >> idx) & 1) {
    value0 = _mm256_andnot_pd(value0, _mm256_set1_pd(-0.0));
    value1 = _mm256_andnot_pd(value1, _mm256_set1_pd(-0.0));
  }
  if ((neg >> idx) & 1) {
    value0 = _mm256_xor_pd(value0, _mm256_set1_pd(-0.0));
    value1 = _mm256_xor_pd(value1, _mm256_set1_pd(-0.0));
  }
  return m256dx2(value0, value1);
}

static uint64_t get_scalar_operand_u64_e64(int addr, const int32_t *sgprs) {
  switch (addr) {
    case 0 ... 127:
      return (static_cast<uint64_t>(sgprs[addr]) & 0xFFFFFFFF) | (static_cast<uint64_t>(sgprs[addr + 1]) << 32);
    case 128:
      return 0;
    case 129 ... 192:
      return static_cast<uint64_t>(addr - 128);
    case 193 ... 208:
      return static_cast<uint64_t>(-((addr - 192)));
    case 240:
      return 0x3fe0000000000000;  // 0.5
    case 241:
      return 0xbfe0000000000000;  // -0.5
    case 242:
      return 0x3ff0000000000000;  // 1.0
    case 243:
      return 0xbff0000000000000;  // -1.0
    case 244:
      return 0x4000000000000000;  // 2.0
    case 245:
      return 0xc000000000000000;  // -2.0
    case 246:
      return 0x4010000000000000;  // 4.0
    case 247:
      return 0xc010000000000000;  // -4.0
    case 248:
      return 0x3fc45f306dc8bdc4;  // 1/(2*PI)
    default:
      throw std::runtime_error("Invalid address for get_scalar_operand_u64_e64");
  }
}

static double get_scalar_operand_e64_f64(int addr, const int32_t *sgprs, const int32_t *literal_constant) {
  union {
    uint64_t u64;
    double f64;
  } value;

  if (addr == 255) {
    value.u64 = static_cast<uint64_t>(literal_constant[0]) << 32;
  } else {
    value.u64 = get_scalar_operand_u64_e64(addr, sgprs);
  }
  return value.f64;
}

static m256dx2 omod_clamp(const m256dx2 value, uint8_t omod, bool clamp) {
  auto value0 = value.value0;
  auto value1 = value.value1;
  if (omod == 1) {
    value0 = _mm256_mul_pd(value0, _mm256_set1_pd(2.0));
    value1 = _mm256_mul_pd(value1, _mm256_set1_pd(2.0));
  } else if (omod == 2) {
    value0 = _mm256_mul_pd(value0, _mm256_set1_pd(4.0));
    value1 = _mm256_mul_pd(value1, _mm256_set1_pd(4.0));
  } else if (omod == 3) {
    value0 = _mm256_mul_pd(value0, _mm256_set1_pd(0.5));
    value1 = _mm256_mul_pd(value1, _mm256_set1_pd(0.5));
  }

  if (clamp) {
    const auto zero = _mm256_setzero_pd();
    const auto one = _mm256_set1_pd(1.0);
    value0 = _mm256_max_pd(zero, _mm256_min_pd(value0, one));
    value1 = _mm256_max_pd(zero, _mm256_min_pd(value1, one));
  }

  return m256dx2(value0, value1);
}

static m256dx2 load_vgpr(int elem, int idx, const int32_t *vgprs) {
  const auto value_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&vgprs[(idx * 32 + elem)]));
  const auto value_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&vgprs[(idx * 32 + elem + 32)]));
  const auto value0 = _mm256_castsi256_pd(_mm256_unpacklo_epi32(value_lo, value_hi));
  const auto value1 = _mm256_castsi256_pd(_mm256_unpackhi_epi32(value_lo, value_hi));
  return m256dx2(value0, value1);
}

static void store_vgpr(int elem, int idx, const m256dx2 value, int32_t *vgprs, __m256i mask) {
  const auto value0 = value.value0;
  const auto value1 = value.value1;
  const auto value0_ps = _mm256_castpd_ps(value0);
  const auto value1_ps = _mm256_castpd_ps(value1);
  const auto value_lo = _mm256_castps_si256(_mm256_shuffle_ps(value0_ps, value1_ps, 0x88));
  const auto value_hi = _mm256_castps_si256(_mm256_shuffle_ps(value0_ps, value1_ps, 0xDD));

  _mm256_maskstore_epi32(&vgprs[(idx * 32 + elem)], mask, value_lo);
  _mm256_maskstore_epi32(&vgprs[(idx * 32 + elem + 32)], mask, value_hi);
}

void v_fma_f64(uint16_t d, uint16_t s0, uint16_t s1, uint16_t s2, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod,
               const int32_t *sgprs, int32_t *vgprs, const int32_t *literal_constant) {
  const auto bitpos = _mm256_set_epi32(1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0);

  for (int32_t elem = 0; elem < 32; elem += 8) {
    const auto i = static_cast<int32_t>(sgprs[126] >> elem);
    const auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

    m256dx2 s0_value, s1_value, s2_value;

    if (s0 >= 256) {
      s0_value = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    if (s1 >= 256) {
      s1_value = load_vgpr(elem, s1 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s1, sgprs, literal_constant);
      s1_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    if (s2 >= 256) {
      s2_value = load_vgpr(elem, s2 - 256, vgprs);
    } else {
      double value = get_scalar_operand_e64_f64(s2, sgprs, literal_constant);
      s2_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    s0_value = abs_neg<0>(s0_value, abs, neg);
    s1_value = abs_neg<1>(s1_value, abs, neg);
    s2_value = abs_neg<2>(s2_value, abs, neg);

    const auto d_value0 = _mm256_fmadd_pd(s0_value.value0, s1_value.value0, s2_value.value0);
    const auto d_value1 = _mm256_fmadd_pd(s0_value.value1, s1_value.value1, s2_value.value1);

    const auto d_value = omod_clamp(m256dx2(d_value0, d_value1), omod, clamp);

    store_vgpr(elem, d, d_value, vgprs, mask);
  }
}

void v_add_f64_e64(uint16_t d, uint16_t s0, uint16_t s1, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod,
                   const int32_t *sgprs, int32_t *vgprs, const int32_t *literal_constant) {
  const auto bitpos = _mm256_set_epi32(1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0);

  for (int32_t elem = 0; elem < 32; elem += 8) {
    const auto i = static_cast<int32_t>(sgprs[126] >> elem);
    const auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

    m256dx2 s0_value, s1_value;

    if (s0 >= 256) {
      s0_value = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    if (s1 >= 256) {
      s1_value = load_vgpr(elem, s1 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s1, sgprs, literal_constant);
      s1_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    s0_value = abs_neg<0>(s0_value, abs, neg);
    s1_value = abs_neg<1>(s1_value, abs, neg);

    const auto d_value0 = _mm256_add_pd(s0_value.value0, s1_value.value0);
    const auto d_value1 = _mm256_add_pd(s0_value.value1, s1_value.value1);

    const auto d_value = omod_clamp(m256dx2(d_value0, d_value1), omod, clamp);

    store_vgpr(elem, d, d_value, vgprs, mask);
  }
}

void v_mul_f64_e32(uint16_t d, uint16_t s0, uint16_t s1, const int32_t *sgprs, int32_t *vgprs,
                   const int32_t *literal_constant) {
  const auto bitpos = _mm256_set_epi32(1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0);

  for (int32_t elem = 0; elem < 32; elem += 8) {
    const auto i = static_cast<int32_t>(sgprs[126] >> elem);
    const auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

    m256dx2 s0_value;

    if (s0 >= 256) {
      s0_value = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value = m256dx2(_mm256_set1_pd(value), _mm256_set1_pd(value));
    }

    const auto s1_value = load_vgpr(elem, s1, vgprs);

    const auto d_value0 = _mm256_mul_pd(s0_value.value0, s1_value.value0);
    const auto d_value1 = _mm256_mul_pd(s0_value.value1, s1_value.value1);

    const auto d_value = m256dx2(d_value0, d_value1);

    store_vgpr(elem, d, d_value, vgprs, mask);
  }
}
