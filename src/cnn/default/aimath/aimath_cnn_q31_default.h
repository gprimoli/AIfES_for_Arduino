#ifndef AIMATH_CNN_Q31_DEFAULT_H
#define AIMATH_CNN_Q31_DEFAULT_H

#include <stdint.h>
#include "core/aifes_core.h"
#include "basic/base/aimath/aimath_q31.h"
#include "basic/default/aimath/aimath_q31_default.h"

void aimath_q31_default_conv2d_fwd(
                    const aitensor_t *input,
                    const uint16_t stride[2],
                    const uint16_t dilation[2],
                    const uint16_t padding[2],
                    const aitensor_t *weights,
                    const aitensor_t *bias,
                    int8_t channel_axis,
                    void *work_space,
                    aitensor_t *output);

void aimath_q31_default_conv2d_bwd(
                    const aitensor_t *x_in,
                    const uint16_t stride[2],
                    const uint16_t dilation[2],
                    const uint16_t padding[2],
                    const aitensor_t *delta_out,
                    int8_t channel_axis,
                    void *work_space,
                    aitensor_t *d_weights);

#endif // AIMATH_CNN_Q31_DEFAULT_H
