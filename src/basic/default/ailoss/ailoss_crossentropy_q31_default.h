/**
 * \file basic/default/ailoss/ailoss_crossentropy_q31_default.h
 */
#ifndef AILOSS_CROSSENTROPY_Q31_DEFAULT
#define AILOSS_CROSSENTROPY_Q31_DEFAULT

#include "basic/base/ailoss/ailoss_crossentropy.h"
#include "basic/default/aimath/aimath_q31_default.h"

typedef struct ailoss_crossentropy ailoss_crossentropy_q31_t;

ailoss_t *ailoss_crossentropy_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

ailoss_t *ailoss_crossentropy_sum_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

ailoss_t *ailoss_crossentropy_mean_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

ailoss_t *ailoss_crossentropy_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

ailoss_t *ailoss_crossentropy_sum_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

ailoss_t *ailoss_crossentropy_mean_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer);

#endif // AILOSS_CROSSENTROPY_Q31_DEFAULT
