/**
 * \file basic/default/aiopti/aiopti_adam_default.c
 * \version 2.2.0
 * \date 12.11.2020
 * \copyright  Copyright (C) 2020-2023  Fraunhofer Institute for Microelectronic Circuits and Systems.
    All rights reserved.<br><br>
    AIfES is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.<br><br>
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.<br><br>
    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *
 * \brief
 * \details
 */

#include "basic/default/aiopti/aiopti_adam_default.h"

aiopti_t *aiopti_adam_f32_default(aiopti_adam_f32_t *opti)
{
	opti->base.base.dtype = aif32;

	opti->base.base.learning_rate = &opti->learning_rate;

	opti->base.beta1 = &opti->beta1;
	opti->base.beta2 = &opti->beta2;
	opti->base.eps = &opti->eps;

	opti->base.beta1t = &opti->beta1t;
	opti->base.beta2t = &opti->beta2t;
	opti->base.one_minus_beta1 = &opti->one_minus_beta1;
	opti->base.one_minus_beta2 = &opti->one_minus_beta2;
	opti->base.lrt = &opti->lrt;

	// ToDo: Check Initialization
	opti->beta1t = opti->beta1;
	opti->beta2t = opti->beta2;
	opti->one_minus_beta1 = 1.0f - opti->beta1;
	opti->one_minus_beta2 = 1.0f - opti->beta2;

	// Set f32 function of optimizer base function begin_step() and end_step()
	opti->base.base.begin_step = aiopti_adam_f32_default_begin_step;
	opti->base.base.end_step = aiopti_adam_f32_default_end_step;

	// Set f32 math functions of adam optimizer
	opti->base.multiply = aimath_f32_default_multiply;
	opti->base.divide = aimath_f32_default_divide;
	opti->base.tensor_add = aimath_f32_default_tensor_add;
	opti->base.tensor_sub = aimath_f32_default_tensor_sub;
	opti->base.scalar_mul = aimath_f32_default_scalar_mul;
	opti->base.scalar_add = aimath_f32_default_scalar_add;
	opti->base.sqrt = aimath_f32_default_sqrt;

	opti->base.zero_tensor = aimath_f32_default_zero_tensor;

	return aiopti_adam(&opti->base);
}

aiopti_t *aiopti_adam_q31_default(aiopti_adam_q31_t *opti)
{
        opti->base.base.dtype = aiq31;

        opti->base.base.learning_rate = &opti->learning_rate;

        opti->base.beta1 = &opti->beta1;
        opti->base.beta2 = &opti->beta2;
        opti->base.eps = &opti->eps;

        opti->base.beta1t = &opti->beta1t;
        opti->base.beta2t = &opti->beta2t;
        opti->base.one_minus_beta1 = &opti->one_minus_beta1;
        opti->base.one_minus_beta2 = &opti->one_minus_beta2;
        opti->base.lrt = &opti->lrt;

        opti->beta1t = opti->beta1;
        opti->beta2t = opti->beta2;
        opti->one_minus_beta1.value = ((int32_t)1 << opti->beta1.shift) + opti->beta1.zero_point - opti->beta1.value;
        opti->one_minus_beta1.shift = opti->beta1.shift;
        opti->one_minus_beta1.zero_point = opti->beta1.zero_point;
        opti->one_minus_beta2.value = ((int32_t)1 << opti->beta2.shift) + opti->beta2.zero_point - opti->beta2.value;
        opti->one_minus_beta2.shift = opti->beta2.shift;
        opti->one_minus_beta2.zero_point = opti->beta2.zero_point;
        opti->lrt.shift = opti->learning_rate.shift;
        opti->lrt.zero_point = opti->learning_rate.zero_point;
        opti->lrt.value = opti->learning_rate.value;

        opti->base.base.begin_step = aiopti_adam_q31_default_begin_step;
        opti->base.base.end_step = aiopti_adam_q31_default_end_step;

        opti->base.multiply = aimath_q31_default_multiply;
        opti->base.divide = aimath_q31_default_divide;
        opti->base.tensor_add = aimath_q31_default_tensor_add_same_shift;
        opti->base.tensor_sub = aimath_q31_default_tensor_sub_different_shift;
        opti->base.scalar_mul = aimath_q31_default_scalar_mul;
        opti->base.scalar_add = aimath_q31_default_scalar_add;
        opti->base.sqrt = aimath_q31_default_tensor_sqrt;

        opti->base.zero_tensor = aimath_q31_default_zero_tensor;

        return aiopti_adam(&opti->base);
}

void aiopti_adam_f32_default_begin_step(aiopti_t *self)
{
	aiopti_adam_t *opti = (aiopti_adam_t *)(self->optimizer_configuration);

	// lr_t = lr * sqrt(1-beta2^t) / (1-beta1^t)
	*((float *) opti->lrt) = *((float *) self->learning_rate) * sqrtf(1.0f - *((float *) opti->beta2t)) / (1.0f - *((float *) opti->beta1t));
	return;
}

void aiopti_adam_f32_default_end_step(aiopti_t *self)
{
	aiopti_adam_t *opti = (aiopti_adam_t *)(self->optimizer_configuration);

	// beta^t <= beta^t * beta;
	*((float *) opti->beta1t) = *((float *) opti->beta1t) * *((float *) opti->beta1);
	*((float *) opti->beta2t) = *((float *) opti->beta2t) * *((float *) opti->beta2);
	*((float *) opti->one_minus_beta1) = 1.0f - *((float *) opti->beta1);
	*((float *) opti->one_minus_beta2) = 1.0f - *((float *) opti->beta2);

        return;
}

void aiopti_adam_q31_default_begin_step(aiopti_t *self)
{
        aiopti_adam_q31_t *opti = (aiopti_adam_q31_t *)(self->optimizer_configuration);

        float lr = Q31_TO_FLOAT(opti->learning_rate.value, opti->learning_rate.shift, opti->learning_rate.zero_point);
        float b1t = Q31_TO_FLOAT(opti->beta1t.value, opti->beta1t.shift, opti->beta1t.zero_point);
        float b2t = Q31_TO_FLOAT(opti->beta2t.value, opti->beta2t.shift, opti->beta2t.zero_point);
        float lr_t = lr * sqrtf(1.0f - b2t) / (1.0f - b1t);
        opti->lrt.value = FLOAT_TO_Q31(lr_t, opti->lrt.shift, opti->lrt.zero_point);
        return;
}

void aiopti_adam_q31_default_end_step(aiopti_t *self)
{
        aiopti_adam_q31_t *opti = (aiopti_adam_q31_t *)(self->optimizer_configuration);

        float b1t = Q31_TO_FLOAT(opti->beta1t.value, opti->beta1t.shift, opti->beta1t.zero_point) *
                     Q31_TO_FLOAT(opti->beta1.value, opti->beta1.shift, opti->beta1.zero_point);
        float b2t = Q31_TO_FLOAT(opti->beta2t.value, opti->beta2t.shift, opti->beta2t.zero_point) *
                     Q31_TO_FLOAT(opti->beta2.value, opti->beta2.shift, opti->beta2.zero_point);
        opti->beta1t.value = FLOAT_TO_Q31(b1t, opti->beta1t.shift, opti->beta1t.zero_point);
        opti->beta2t.value = FLOAT_TO_Q31(b2t, opti->beta2t.shift, opti->beta2t.zero_point);

        opti->one_minus_beta1.value = ((int32_t)1 << opti->beta1.shift) + opti->beta1.zero_point - opti->beta1.value;
        opti->one_minus_beta2.value = ((int32_t)1 << opti->beta2.shift) + opti->beta2.zero_point - opti->beta2.value;

        return;
}
