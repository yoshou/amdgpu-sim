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

template <size_t idx>
static std::tuple<__m256d, __m256d> abs_neg(__m256d value0, __m256d value1, uint8_t abs, uint8_t neg) {
  if ((abs >> idx) & 1) {
    value0 = _mm256_andnot_pd(value0, _mm256_set1_pd(-0.0));
    value1 = _mm256_andnot_pd(value1, _mm256_set1_pd(-0.0));
  }
  if ((neg >> idx) & 1) {
    value0 = _mm256_xor_pd(value0, _mm256_set1_pd(-0.0));
    value1 = _mm256_xor_pd(value1, _mm256_set1_pd(-0.0));
  }
  return std::make_tuple(value0, value1);
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
  if (addr == 255) {
    const auto value = static_cast<uint64_t>(literal_constant[0]) << 32;
    return *reinterpret_cast<const double *>(&value);
  } else {
    const auto value = get_scalar_operand_u64_e64(addr, sgprs);
    return *reinterpret_cast<const double *>(&value);
  }
}

static std::tuple<__m256d, __m256d> omod_clamp(__m256d value0, __m256d value1, uint8_t omod, bool clamp) {
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

  return std::make_tuple(value0, value1);
}

static std::tuple<__m256d, __m256d> load_vgpr(int elem, int idx, const int32_t *vgprs) {
  const auto value_lo = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&vgprs[(idx * 32 + elem)]));
  const auto value_hi = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(&vgprs[(idx * 32 + elem + 32)]));
  const auto value0 = _mm256_castsi256_pd(_mm256_unpacklo_epi32(value_lo, value_hi));
  const auto value1 = _mm256_castsi256_pd(_mm256_unpackhi_epi32(value_lo, value_hi));
  return std::make_tuple(value0, value1);
}

static void store_vgpr(int elem, int idx, __m256d value0, __m256d value1, int32_t *vgprs, __m256i mask) {
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

    __m256d s0_value0, s0_value1, s1_value0, s1_value1, s2_value0, s2_value1;

    if (s0 >= 256) {
      std::tie(s0_value0, s0_value1) = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value0 = _mm256_set1_pd(value);
      s0_value1 = _mm256_set1_pd(value);
    }

    if (s1 >= 256) {
      std::tie(s1_value0, s1_value1) = load_vgpr(elem, s1 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s1, sgprs, literal_constant);
      s1_value0 = _mm256_set1_pd(value);
      s1_value1 = _mm256_set1_pd(value);
    }

    if (s2 >= 256) {
      std::tie(s2_value0, s2_value1) = load_vgpr(elem, s2 - 256, vgprs);
    } else {
      double value = get_scalar_operand_e64_f64(s2, sgprs, literal_constant);
      s2_value0 = _mm256_set1_pd(value);
      s2_value1 = _mm256_set1_pd(value);
    }

    std::tie(s0_value0, s0_value1) = abs_neg<0>(s0_value0, s0_value1, abs, neg);
    std::tie(s1_value0, s1_value1) = abs_neg<1>(s1_value0, s1_value1, abs, neg);
    std::tie(s2_value0, s2_value1) = abs_neg<2>(s2_value0, s2_value1, abs, neg);

    auto d_value0 = _mm256_fmadd_pd(s0_value0, s1_value0, s2_value0);
    auto d_value1 = _mm256_fmadd_pd(s0_value1, s1_value1, s2_value1);

    std::tie(d_value0, d_value1) = omod_clamp(d_value0, d_value1, omod, clamp);

    store_vgpr(elem, d, d_value0, d_value1, vgprs, mask);
  }
}

void v_add_f64_e64(uint16_t d, uint16_t s0, uint16_t s1, uint8_t abs, uint8_t neg, bool clamp, uint8_t omod,
                   const int32_t *sgprs, int32_t *vgprs, const int32_t *literal_constant) {
  const auto bitpos = _mm256_set_epi32(1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0);

  for (int32_t elem = 0; elem < 32; elem += 8) {
    const auto i = static_cast<int32_t>(sgprs[126] >> elem);
    const auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

    __m256d s0_value0, s0_value1, s1_value0, s1_value1;

    if (s0 >= 256) {
      std::tie(s0_value0, s0_value1) = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value0 = _mm256_set1_pd(value);
      s0_value1 = _mm256_set1_pd(value);
    }

    if (s1 >= 256) {
      std::tie(s1_value0, s1_value1) = load_vgpr(elem, s1 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s1, sgprs, literal_constant);
      s1_value0 = _mm256_set1_pd(value);
      s1_value1 = _mm256_set1_pd(value);
    }

    std::tie(s0_value0, s0_value1) = abs_neg<0>(s0_value0, s0_value1, abs, neg);
    std::tie(s1_value0, s1_value1) = abs_neg<1>(s1_value0, s1_value1, abs, neg);

    auto d_value0 = _mm256_add_pd(s0_value0, s1_value0);
    auto d_value1 = _mm256_add_pd(s0_value1, s1_value1);

    std::tie(d_value0, d_value1) = omod_clamp(d_value0, d_value1, omod, clamp);

    store_vgpr(elem, d, d_value0, d_value1, vgprs, mask);
  }
}

void v_mul_f64_e32(uint16_t d, uint16_t s0, uint16_t s1, const int32_t *sgprs, int32_t *vgprs,
                   const int32_t *literal_constant) {
  const auto bitpos = _mm256_set_epi32(1 << 7, 1 << 6, 1 << 5, 1 << 4, 1 << 3, 1 << 2, 1 << 1, 1 << 0);

  for (int32_t elem = 0; elem < 32; elem += 8) {
    const auto i = static_cast<int32_t>(sgprs[126] >> elem);
    const auto mask = _mm256_cmpeq_epi32(_mm256_and_si256(_mm256_set1_epi32(i), bitpos), bitpos);

    __m256d s0_value0, s0_value1, s1_value0, s1_value1;

    if (s0 >= 256) {
      std::tie(s0_value0, s0_value1) = load_vgpr(elem, s0 - 256, vgprs);
    } else {
      const auto value = get_scalar_operand_e64_f64(s0, sgprs, literal_constant);
      s0_value0 = _mm256_set1_pd(value);
      s0_value1 = _mm256_set1_pd(value);
    }

    std::tie(s1_value0, s1_value1) = load_vgpr(elem, s1, vgprs);

    auto d_value0 = _mm256_mul_pd(s0_value0, s1_value0);
    auto d_value1 = _mm256_mul_pd(s0_value1, s1_value1);

    store_vgpr(elem, d, d_value0, d_value1, vgprs, mask);
  }
}
