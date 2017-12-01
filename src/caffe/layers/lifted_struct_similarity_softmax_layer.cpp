#include <algorithm>
#include <vector>

#include "caffe/custom_layers.hpp"

namespace caffe
{

    template<typename Dtype>
    void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                                                   const vector<Blob<Dtype> *> &top)
    {
        LossLayer<Dtype>::LayerSetUp(bottom, top);
        // Ensure that the feature vectors are only 1-dimensional (N feature vectors)
        CHECK_EQ(bottom[0]->height(), 1);
        CHECK_EQ(bottom[0]->width(), 1);
        // Ensure that the labels are singletons
        CHECK_EQ(bottom[1]->channels(), 1);
        CHECK_EQ(bottom[1]->height(), 1);
        CHECK_EQ(bottom[1]->width(), 1);
        // Ensure that the similarity vectors are 1-dimensional
        CHECK_EQ(bottom[2]->height(), 1);
        CHECK_EQ(bottom[2]->width(), 1);
        // List of member variables defined in /include/caffe/loss_layers.hpp
        //   diff_, dist_sq_, summer_vec_, loss_aug_inference_

        dist_sq_.Reshape(bottom[0]->num(), 1, 1, 1);
        dot_.Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
        ones_.Reshape(bottom[0]->num(), 1, 1, 1);  // n by 1 vector of ones.
        for (int i = 0; i < bottom[0]->num(); ++i)
        {
            ones_.mutable_cpu_data()[i] = Dtype(1);
        }
        blob_pos_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
        blob_neg_diff_.Reshape(bottom[0]->channels(), 1, 1, 1);
        iteration = 0;
    }

    template<typename Dtype>
    void
    LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                                               const vector<Blob<Dtype> *> &top)
    {
        iteration++;
        const int channels = bottom[0]->channels();
        for (int i = 0; i < bottom[0]->num(); i++)
        {
            dist_sq_.mutable_cpu_data()[i] = caffe_cpu_dot(channels, bottom[0]->cpu_data() + (i * channels),
                                                           bottom[0]->cpu_data() + (i * channels));
        }

        int M_ = bottom[0]->num(); // batch size N
        int N_ = bottom[0]->num(); // batch size N
        int K_ = bottom[0]->channels(); // number of channels K or the embedding size

        int L_ = bottom[2]->channels(); // maximum number of similar views + 1 for the sample label

        const Dtype *bottom_data1 = bottom[0]->cpu_data();
        const Dtype *bottom_data2 = bottom[0]->cpu_data();

        for (int i = 0; i < bottom[0]->count(); i++)
        {
            CHECK_EQ(isnan(bottom_data1[i]), false);
        }

        // Compute the pairwise euclidean distance matrix D (dot)
        caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, Dtype(-2), bottom_data1, bottom_data2, Dtype(0),
                              dot_.mutable_cpu_data());

        // add ||x_i||^2 to all elements in row i
        for (int i = 0; i < N_; i++)
        {
            caffe_axpy(N_, dist_sq_.cpu_data()[i], ones_.cpu_data(), dot_.mutable_cpu_data() + i * N_);
        }

        // add the norm vector to row i
        for (int i = 0; i < N_; i++)
        {
            caffe_axpy(N_, Dtype(1.0), dist_sq_.cpu_data(), dot_.mutable_cpu_data() + i * N_);
            for (int j = 0; j < N_; j++)
            {
                if (dot_.mutable_cpu_data()[i * N_ + j] < 0 || isnan(dot_.mutable_cpu_data()[i * N_ + j]))
                {
                    dot_.mutable_cpu_data()[i * N_ + j] = 0;
                }
            }
        }

        // construct pairwise label matrix
        bool any_pair_found = false;
        vector<vector<bool> > label_mat(N_, vector<bool>(N_, false));
        for (int i = 0; i < N_; i++)
        {
            int id_1 = bottom[1]->cpu_data()[i];
            int idx_1_offset = id_1 * L_;
            int id_1_ref = bottom[2]->cpu_data()[idx_1_offset + L_ - 1];
            CHECK_EQ(id_1, id_1_ref) << "iteration: " << iteration << " batch index: " << i
                                     << " ID1 " << id_1 << " != " << id_1_ref;

            label_mat[i][i] = true; // write ones on the diagonal

            for (int j = i + 1; j < N_; j++)
            {
                int id_2 = bottom[1]->cpu_data()[j];
                int idx_2_offset = id_2 * L_;
                int id_2_ref = bottom[2]->cpu_data()[idx_2_offset + L_ - 1];
                CHECK_EQ(id_2, id_2_ref) << "iteration: " << iteration << " batch index: " << j
                                         << " ID2 " << id_2 << " != " << id_2_ref;

                // loop over all similar views, excluding the last position (self index)
                for (int k = 0; k < L_ - 1; k++)
                {
                    int sw = bottom[2]->cpu_data()[idx_2_offset + k];
                    if (sw == 0) // end of the list of similar views
                    {
                        break;
                    } else if (sw == id_1)
                    {
                        any_pair_found = true;
                        label_mat[i][j] = true;
                        label_mat[j][i] = true;
                        break;
                    }
                }
            }
        }
        CHECK_EQ(any_pair_found, true) << "No similar pair found in batch " << iteration;

        Dtype margin = this->layer_param_.lifted_struct_sim_softmax_loss_param().margin();

        Dtype loss(0.0);
        num_constraints = Dtype(0.0);
        // the gradient will also be stored in bottom[0]
        // apparently, referencing bottom[0] with const accessor forks the data, i.e. changes written to bout[i] are not
        // propagated to bin[i]
        const Dtype *bin = bottom[0]->cpu_data();
        Dtype *bout = bottom[0]->mutable_cpu_diff();

        // zero initialize bottom[0]->mutable_cpu_diff();
        for (int i = 0; i < N_; i++)
        {
            caffe_set(K_, Dtype(0.0), bout + i * K_);
        }

        // loop upper triangular matrix and look for positive anchors
        for (int i = 0; i < N_; i++)
        {
            for (int j = i + 1; j < N_; j++)
            {
                // if this is a negative pair @ anchor (i, j), skip
                if (!label_mat[i][j])
                {
                    continue;
                }
                // it is a positive pair @ anchor (i, j)

                Dtype dist_pos = sqrt(dot_.cpu_data()[i * N_ + j]); // D_i,j^2
                CHECK_EQ(isnan(dist_pos), false);

                // what does this do? Subtract 0 from 0?
                caffe_sub(K_, bin + i * K_, bin + j * K_, blob_pos_diff_.mutable_cpu_data());

                // 1. count the number of negatives for this positive
                int num_negatives = 0;
                for (int k = 0; k < N_; k++)
                {
                    if (!label_mat[i][k]) // for sample i
                    {
                        num_negatives += 1;
                    }
                    if (!label_mat[j][k]) // for sample j
                    {
                        num_negatives += 1;
                    }
                }

                loss_aug_inference_.Reshape(num_negatives, 1, 1, 1);

                // vector of ones used to sum along channels
                summer_vec_.Reshape(num_negatives, 1, 1, 1);
                caffe_set(num_negatives, Dtype(1), summer_vec_.mutable_cpu_data());

                // 2. compute loss augmented inference
                int neg_idx = 0;
                // mine negative (anchor i, neg k)
                for (int k = 0; k < N_; k++)
                {
                    if (!label_mat[i][k])
                    {
                        Dtype v = margin - sqrt(dot_.cpu_data()[i * N_ + k]);
                        CHECK_EQ(isnan(v), false);
                        loss_aug_inference_.mutable_cpu_data()[neg_idx] = v;

                        neg_idx++;
                    }
                }

                // mine negative (anchor j, neg k)
                for (int k = 0; k < N_; k++)
                {
                    if (!label_mat[j][k])
                    {
                        Dtype v = margin - sqrt(dot_.cpu_data()[j * N_ + k]);
                        CHECK_EQ(isnan(v), false);
                        loss_aug_inference_.mutable_cpu_data()[neg_idx] = v;

                        neg_idx++;
                    }
                }

                // compute softmax of loss aug inference vector

                Dtype max_elem = *std::max_element(loss_aug_inference_.cpu_data(),
                                                   loss_aug_inference_.cpu_data() + num_negatives);
                CHECK_EQ(isnan(max_elem), false);
                // shift the argument of the exponent towards 0 for numeric stability, see
                // https://hips.seas.harvard.edu/blog/2013/01/09/computing-log-sum-exp/
                caffe_add_scalar(loss_aug_inference_.count(), -max_elem, loss_aug_inference_.mutable_cpu_data());
                caffe_exp(loss_aug_inference_.count(), loss_aug_inference_.mutable_cpu_data(),
                          loss_aug_inference_.mutable_cpu_data());
                Dtype softmax_arg = caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(),
                                                  loss_aug_inference_.mutable_cpu_data()); // why use mutable cpu data here?
                CHECK_EQ(isnan(softmax_arg), false);
                // TODO: shift it back again
                Dtype soft_maximum = log(softmax_arg) + max_elem;
                CHECK_EQ(isnan(soft_maximum), false);

                // hinge J = soft_maximum + D_ij (positive pair similarity)
                Dtype this_loss = std::max(soft_maximum + dist_pos, Dtype(0.0));
                CHECK_EQ(isnan(this_loss), false);

                // squared hinge
                loss += this_loss * this_loss;
                CHECK_EQ(isnan(loss), false);

                num_constraints += Dtype(1.0);

                // 3. compute gradients
                Dtype sum_exp = caffe_cpu_dot(num_negatives, summer_vec_.cpu_data(),
                                              loss_aug_inference_.mutable_cpu_data());
                CHECK_EQ(isnan(sum_exp), false);

                // update from positive distance dJ_dD_{ij}; update x_i, x_j

                Dtype scaler = Dtype(2.0) * this_loss / dist_pos;
                CHECK_EQ(isnan(scaler), false);

                // update x_i
                caffe_axpy(K_, scaler * Dtype(1.0), blob_pos_diff_.cpu_data(), bout + i * K_);
                // update x_j
                caffe_axpy(K_, scaler * Dtype(-1.0), blob_pos_diff_.cpu_data(), bout + j * K_);

                // update from negative distance dJ_dD_{ik}; update x_i, x_k
                neg_idx = 0;
                Dtype dJ_dDik(0.0);
                for (int k = 0; k < N_; k++)
                {
                    if (!label_mat[i][k])
                    {
                        caffe_sub(K_, bin + i * K_, bin + k * K_, blob_neg_diff_.mutable_cpu_data());

                        dJ_dDik = Dtype(-2.0) * this_loss * loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDik / sqrt(dot_.cpu_data()[i * N_ + k]); // -inf???

                        // TODO: why is dot[i*N+k] == 0 ? because we set it before? Why is this then a negative pair? verify indices with ground truth in matlab
                        CHECK_EQ(isnan(scaler), false);
                        CHECK_EQ(isinf(scaler), false);

                        // update x_i
                        caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout + i * K_);
                        CHECK_EQ(isnan(bout[i * K_]), false);
                        // update x_k
                        caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout + k * K_);
                        CHECK_EQ(isnan(bout[k * K_]), false);
                    }
                }

                // update from negative distance dJ_dD_{jk}; update x_j, x_k
                Dtype dJ_dDjk(0.0);
                for (int k = 0; k < N_; k++)
                {
                    if (!label_mat[j][k])
                    {
                        caffe_sub(K_, bin + j * K_, bin + k * K_, blob_neg_diff_.mutable_cpu_data());

                        dJ_dDjk = Dtype(-2.0) * this_loss * loss_aug_inference_.cpu_data()[neg_idx] / sum_exp;
                        neg_idx++;

                        scaler = dJ_dDjk / sqrt(dot_.cpu_data()[j * N_ + k]);
                        CHECK_EQ(isnan(scaler), false);
                        CHECK_EQ(isinf(scaler), false);

                        // update x_j
                        caffe_axpy(K_, scaler * Dtype(1.0), blob_neg_diff_.cpu_data(), bout + j * K_);
                        CHECK_EQ(isnan(bout[j * K_]), false);
                        // update x_k
                        caffe_axpy(K_, scaler * Dtype(-1.0), blob_neg_diff_.cpu_data(), bout + k * K_);
                        CHECK_EQ(isnan(bout[k * K_]), false);
                    }
                }
            }
        }
        CHECK_GE(num_constraints, 1);

        loss = loss / num_constraints / Dtype(2.0);
        CHECK_EQ(isnan(loss), false);

        top[0]->mutable_cpu_data()[0] = loss;
    }

    template<typename Dtype>
    void LiftedStructSimilaritySoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype> *> &top,
                                                                     const vector<bool> &propagate_down,
                                                                     const vector<Blob<Dtype> *> &bottom)
    {
        const Dtype alpha = top[0]->cpu_diff()[0] / num_constraints / Dtype(2.0);
        CHECK_EQ(isnan(alpha), false);

        int num = bottom[0]->num();
        int channels = bottom[0]->channels();
        for (int i = 0; i < num; i++)
        {
            Dtype *bout = bottom[0]->mutable_cpu_diff();
            caffe_scal(channels, alpha, bout + (i * channels));
        }
    }

#ifdef CPU_ONLY
    STUB_GPU(LiftedStructSimilaritySoftmaxLossLayer);
#endif

    INSTANTIATE_CLASS(LiftedStructSimilaritySoftmaxLossLayer);

    REGISTER_LAYER_CLASS(LiftedStructSimilaritySoftmaxLoss);

}  // namespace caffe



