/**
 * \file basic/default/ailoss/ailoss_crossentropy_q31_default.c
 */
#include "basic/default/ailoss/ailoss_crossentropy_q31_default.h"

AISTRING_STORAGE_WRAPPER(aistring_error_loss_crossentropy_q31_1, "[ailoss_crossentropy_q31_default] Input layer type not supported\n");

ailoss_t *ailoss_crossentropy_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    return ailoss_crossentropy_mean_q31_default(loss, input_layer);
}

ailoss_t *ailoss_crossentropy_sum_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    loss->dtype = aiq31;

    loss->tensor_sub = aimath_q31_default_tensor_sub_different_shift;
    loss->scale_by_batch_size = 0;

    if (input_layer->layer_type == ailayer_softmax_type) {
        loss->crossentropy = aimath_q31_default_categorical_crossentropy_sum;
    } else {
#ifdef AIDEBUG_PRINT_ERROR_MESSAGES
        AILOG_E(aistring_error_loss_crossentropy_q31_1);
#endif
        return 0;
    }

    return ailoss_crossentropy(loss, input_layer);
}

ailoss_t *ailoss_crossentropy_mean_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    loss->dtype = aiq31;

    loss->tensor_sub = aimath_q31_default_tensor_sub_different_shift;
    loss->scale_by_batch_size = aimath_q31_default_scale_by_batch_size;

    if (input_layer->layer_type == ailayer_softmax_type) {
        loss->crossentropy = aimath_q31_default_categorical_crossentropy_mean;
    } else {
#ifdef AIDEBUG_PRINT_ERROR_MESSAGES
        AILOG_E(aistring_error_loss_crossentropy_q31_1);
#endif
        return 0;
    }

    return ailoss_crossentropy(loss, input_layer);
}

ailoss_t *ailoss_crossentropy_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    return ailoss_crossentropy_mean_sparse8_q31_default(loss, input_layer);
}

ailoss_t *ailoss_crossentropy_sum_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    loss->dtype = aiq31;

    loss->tensor_sub = aimath_q31_default_tensor_sub_different_shift;
    loss->scale_by_batch_size = 0;

    if (input_layer->layer_type == ailayer_softmax_type) {
        loss->crossentropy = aimath_q31_default_categorical_crossentropy_sum_sparse8;
    } else {
#ifdef AIDEBUG_PRINT_ERROR_MESSAGES
        AILOG_E(aistring_error_loss_crossentropy_q31_1);
#endif
        return 0;
    }

    return ailoss_crossentropy(loss, input_layer);
}

ailoss_t *ailoss_crossentropy_mean_sparse8_q31_default(ailoss_crossentropy_q31_t *loss, ailayer_t *input_layer)
{
    loss->dtype = aiq31;

    loss->tensor_sub = aimath_q31_default_tensor_sub_different_shift;
    loss->scale_by_batch_size = aimath_q31_default_scale_by_batch_size;

    if (input_layer->layer_type == ailayer_softmax_type) {
        loss->crossentropy = aimath_q31_default_categorical_crossentropy_mean_sparse8;
    } else {
#ifdef AIDEBUG_PRINT_ERROR_MESSAGES
        AILOG_E(aistring_error_loss_crossentropy_q31_1);
#endif
        return 0;
    }

    return ailoss_crossentropy(loss, input_layer);
}
