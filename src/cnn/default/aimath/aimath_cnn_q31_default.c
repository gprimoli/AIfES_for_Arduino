#include "cnn/default/aimath/aimath_cnn_q31_default.h"

void aimath_q31_default_conv2d_fwd(
                    const aitensor_t *input,
                    const uint16_t stride[2],
                    const uint16_t dilation[2],
                    const uint16_t padding[2],
                    const aitensor_t *weights,
                    const aitensor_t *bias,
                    int8_t channel_axis,
                    void *work_space,
                    aitensor_t *output)
{
    uint8_t channel_uaxis = channel_axis < 0 ? input->dim + channel_axis : channel_axis;
    uint8_t h_ax, w_ax;
    if(channel_uaxis == 1){
        h_ax = 2; w_ax = 3;
    } else {
        h_ax = 1; w_ax = 2;
    }

    uint16_t N = input->shape[0];
    uint16_t C = input->shape[channel_uaxis];
    uint16_t H = input->shape[h_ax];
    uint16_t W = input->shape[w_ax];

    uint16_t F = weights->shape[0];
    uint16_t K_h = weights->shape[h_ax];
    uint16_t K_w = weights->shape[w_ax];

    uint16_t out_h = output->shape[h_ax];
    uint16_t out_w = output->shape[w_ax];

    int32_t *x_data = (int32_t *)input->data;
    int32_t *w_data = (int32_t *)weights->data;
    int32_t *b_data = bias ? (int32_t *)bias->data : 0;
    int32_t *y_data = (int32_t *)output->data;

    aimath_q31_params_t *x_q = (aimath_q31_params_t *)input->tensor_params;
    aimath_q31_params_t *w_q = (aimath_q31_params_t *)weights->tensor_params;
    aimath_q31_params_t *y_q = (aimath_q31_params_t *)output->tensor_params;

    uint16_t output_shift, final_shift;
    if(y_q->shift > x_q->shift + w_q->shift){
        output_shift = 0;
        final_shift = y_q->shift - x_q->shift - w_q->shift;
    } else {
        output_shift = x_q->shift + w_q->shift - y_q->shift;
        final_shift = 0;
    }

    uint32_t in_n = channel_uaxis==1 ? C*H*W : H*W*C;
    uint32_t in_c = channel_uaxis==1 ? H*W : 1;
    uint32_t in_h = channel_uaxis==1 ? W : W*C;
    uint32_t in_w = channel_uaxis==1 ? 1 : C;

    uint32_t w_f = channel_uaxis==1 ? C*K_h*K_w : K_h*K_w*C;
    uint32_t w_c = channel_uaxis==1 ? K_h*K_w : 1;
    uint32_t w_h = channel_uaxis==1 ? K_w : K_w*C;
    uint32_t w_w = channel_uaxis==1 ? 1 : C;

    uint32_t out_n = channel_uaxis==1 ? F*out_h*out_w : out_h*out_w*F;
    uint32_t out_f = channel_uaxis==1 ? out_h*out_w : 1;
    uint32_t out_hm = channel_uaxis==1 ? out_w : out_w*F;
    uint32_t out_wm = channel_uaxis==1 ? 1 : F;

    aimath_q31_default_init_zeros(output);

    for(uint16_t n=0;n<N;n++){
        for(uint16_t f=0;f<F;f++){
            for(uint16_t oh=0;oh<out_h;oh++){
                for(uint16_t ow=0;ow<out_w;ow++){
                    int64_t sum=0, sum_x=0, sum_wt=0;
                    uint32_t count=0;
                    for(uint16_t c=0;c<C;c++){
                        for(uint16_t kh=0;kh<K_h;kh++){
                            int32_t ih = (int32_t)oh*stride[0] - padding[0] + kh*dilation[0];
                            if(ih < 0 || ih >= H) continue;
                            for(uint16_t kw=0;kw<K_w;kw++){
                                int32_t iw = (int32_t)ow*stride[1] - padding[1] + kw*dilation[1];
                                if(iw < 0 || iw >= W) continue;
                                int32_t x = x_data[n*in_n + c*in_c + ih*in_h + iw*in_w];
                                int32_t wv = w_data[f*w_f + c*w_c + kh*w_h + kw*w_w];
                                sum += ((int64_t)x * wv);
                                sum_x += x;
                                sum_wt += wv;
                                count++;
                            }
                        }
                    }
                    sum -= (int64_t)x_q->zero_point * sum_wt;
                    sum -= (int64_t)w_q->zero_point * sum_x;
                    sum += (int64_t)count * x_q->zero_point * w_q->zero_point;
                    if(b_data) sum += b_data[f];
                    sum = ((sum >> output_shift) << final_shift);
                    sum += y_q->zero_point;
                    y_data[n*out_n + f*out_f + oh*out_hm + ow*out_wm] = (int32_t)sum;
                }
            }
        }
    }
}

void aimath_q31_default_conv2d_bwd(
                    const aitensor_t *x_in,
                    const uint16_t stride[2],
                    const uint16_t dilation[2],
                    const uint16_t padding[2],
                    const aitensor_t *delta_out,
                    int8_t channel_axis,
                    void *work_space,
                    aitensor_t *d_weights)
{
    uint8_t channel_uaxis = channel_axis < 0 ? x_in->dim + channel_axis : channel_axis;
    uint8_t h_ax, w_ax;
    if(channel_uaxis == 1){ h_ax=2; w_ax=3; } else { h_ax=1; w_ax=2; }

    uint16_t N = x_in->shape[0];
    uint16_t C = x_in->shape[channel_uaxis];
    uint16_t H = x_in->shape[h_ax];
    uint16_t W = x_in->shape[w_ax];

    uint16_t F = delta_out->shape[channel_uaxis];
    uint16_t out_h = delta_out->shape[h_ax];
    uint16_t out_w = delta_out->shape[w_ax];

    uint16_t K_h = d_weights->shape[h_ax];
    uint16_t K_w = d_weights->shape[w_ax];

    int32_t *x_data = (int32_t *)x_in->data;
    int32_t *dy_data = (int32_t *)delta_out->data;
    int32_t *dw_data = (int32_t *)d_weights->data;

    aimath_q31_params_t *x_q = (aimath_q31_params_t *)x_in->tensor_params;
    aimath_q31_params_t *dy_q = (aimath_q31_params_t *)delta_out->tensor_params;
    aimath_q31_params_t *dw_q = (aimath_q31_params_t *)d_weights->tensor_params;

    uint16_t output_shift, final_shift;
    if(dw_q->shift > x_q->shift + dy_q->shift){
        output_shift = 0;
        final_shift = dw_q->shift - x_q->shift - dy_q->shift;
    } else {
        output_shift = x_q->shift + dy_q->shift - dw_q->shift;
        final_shift = 0;
    }

    uint32_t x_n = channel_uaxis==1 ? C*H*W : H*W*C;
    uint32_t x_c = channel_uaxis==1 ? H*W : 1;
    uint32_t x_h = channel_uaxis==1 ? W : W*C;
    uint32_t x_w = channel_uaxis==1 ? 1 : C;

    uint32_t dy_n = channel_uaxis==1 ? F*out_h*out_w : out_h*out_w*F;
    uint32_t dy_f = channel_uaxis==1 ? out_h*out_w : 1;
    uint32_t dy_hm = channel_uaxis==1 ? out_w : out_w*F;
    uint32_t dy_wm = channel_uaxis==1 ? 1 : F;

    uint32_t dw_f = channel_uaxis==1 ? C*K_h*K_w : K_h*K_w*C;
    uint32_t dw_c = channel_uaxis==1 ? K_h*K_w : 1;
    uint32_t dw_h = channel_uaxis==1 ? K_w : K_w*C;
    uint32_t dw_w = channel_uaxis==1 ? 1 : C;

    aimath_q31_default_init_zeros(d_weights);

    for(uint16_t f=0; f<F; f++){
        for(uint16_t c=0; c<C; c++){
            for(uint16_t kh=0; kh<K_h; kh++){
                for(uint16_t kw=0; kw<K_w; kw++){
                    int64_t sum=0,sum_x=0,sum_dy=0; uint32_t count=0;
                    for(uint16_t n=0;n<N;n++){
                        for(uint16_t oh=0; oh<out_h; oh++){
                            int32_t ih = (int32_t)oh*stride[0] - padding[0] + kh*dilation[0];
                            if(ih < 0 || ih >= H) continue;
                            for(uint16_t ow=0; ow<out_w; ow++){
                                int32_t iw = (int32_t)ow*stride[1] - padding[1] + kw*dilation[1];
                                if(iw < 0 || iw >= W) continue;
                                int32_t x = x_data[n*x_n + c*x_c + ih*x_h + iw*x_w];
                                int32_t dy = dy_data[n*dy_n + f*dy_f + oh*dy_hm + ow*dy_wm];
                                sum += (int64_t)x * dy;
                                sum_x += x;
                                sum_dy += dy;
                                count++;
                            }
                        }
                    }
                    sum -= (int64_t)x_q->zero_point * sum_dy;
                    sum -= (int64_t)dy_q->zero_point * sum_x;
                    sum += (int64_t)count * x_q->zero_point * dy_q->zero_point;
                    sum = ((sum >> output_shift) << final_shift);
                    sum += dw_q->zero_point;
                    dw_data[f*dw_f + c*dw_c + kh*dw_h + kw*dw_w] = (int32_t)sum;
                }
            }
        }
    }
}
