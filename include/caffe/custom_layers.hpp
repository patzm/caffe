#ifndef CAFFE_CUSTOM_LAYERS_HPP_
#define CAFFE_CUSTOM_LAYERS_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/loss_layer.hpp"
#include "caffe/layers/neuron_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe
{
    template<typename Dtype>
    class AggregateLayer : public Layer<Dtype>
    {
    public:
        explicit AggregateLayer(const LayerParameter &param)
                : Layer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const
        { return "Aggregate"; }

        virtual inline int ExactNumBottomBlobs() const
        { return -1; }

        virtual inline int ExactNumTopBlobs() const
        { return 1; }

        virtual inline int MinTopBlobs() const
        { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);
    };

    /* ROIPoolingLayer - Region of Interest Pooling Layer
    */
    template<typename Dtype>
    class ROIPoolingLayer : public Layer<Dtype>
    {
    public:
        explicit ROIPoolingLayer(const LayerParameter &param)
                : Layer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const
        { return "ROIPooling"; }

        virtual inline int MinBottomBlobs() const
        { return 2; }

        virtual inline int MaxBottomBlobs() const
        { return 2; }

        virtual inline int MinTopBlobs() const
        { return 1; }

        virtual inline int MaxTopBlobs() const
        { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        int channels_;
        int height_;
        int width_;
        int pooled_height_;
        int pooled_width_;
        Dtype spatial_scale_;
        Blob<int> max_idx_;
    };

    template<typename Dtype>
    class SmoothL1LossLayer : public LossLayer<Dtype>
    {
    public:
        explicit SmoothL1LossLayer(const LayerParameter &param)
                : LossLayer<Dtype>(param), diff_()
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const
        { return "SmoothL1Loss"; }

        virtual inline int ExactNumBottomBlobs() const
        { return -1; }

        virtual inline int MinBottomBlobs() const
        { return 2; }

        virtual inline int MaxBottomBlobs() const
        { return 3; }

        /**
        * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
        * to both inputs -- override to return true and always allow force_backward.
        */
        virtual inline bool AllowForceBackward(const int bottom_index) const
        {
            return true;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        Blob<Dtype> diff_;
        Blob<Dtype> errors_;
        bool has_weights_;
    };

    /* Yuanyang adding triplet loss layer */
    /* *
    * * @brief Computes the triplet loss
    * */
    template<typename Dtype>
    class TripletLossLayer : public LossLayer<Dtype>
    {
    public:
        explicit TripletLossLayer(const LayerParameter &param) : LossLayer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual inline int ExactNumBottomBlobs() const
        { return 3; }

        virtual inline const char *type() const
        { return "TripletLoss"; }

        /* *
        * * Unlike most loss layers, in the TripletLossLayer we can back-propagate
        * * to the first three inputs.
        * */
        virtual inline bool AllowForceBackward(const int bottom_index) const
        {
            return bottom_index != 3;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom, const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        Blob<Dtype> diff_ap_;  // cached for backward pass
        Blob<Dtype> diff_an_;  // cached for backward pass
        Blob<Dtype> diff_pn_;  // cached for backward pass

        Blob<Dtype> diff_sq_ap_;  // cached for backward pass
        Blob<Dtype> diff_sq_an_;  // tmp storage for gpu forward pass

        Blob<Dtype> dist_sq_ap_;  // cached for backward pass
        Blob<Dtype> dist_sq_an_;  // cached for backward pass

        Blob<Dtype> summer_vec_;  // tmp storage for gpu forward pass
        Blob<Dtype> dist_binary_;  // tmp storage for gpu forward pass

        static const Dtype sampleW_ = Dtype(0.5);
    };

    template<typename Dtype>
    class LiftedStructSimilaritySoftmaxLossLayer : public LossLayer<Dtype>
    {
    public:
        explicit LiftedStructSimilaritySoftmaxLossLayer(const LayerParameter &param)
                : LossLayer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual inline int ExactNumBottomBlobs() const
        { return 3; }

        virtual inline const char *type() const
        { return "LiftedStructSimilaritySoftmaxLoss"; }

        virtual inline bool AllowForceBackward(const int bottom_index) const
        {
            return bottom_index != 1;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        Blob<Dtype> dist_sq_;  // cached for backward pass
        Blob<Dtype> dot_;
        Blob<Dtype> ones_;
        Blob<Dtype> blob_pos_diff_;
        Blob<Dtype> blob_neg_diff_;
        Blob<Dtype> loss_aug_inference_;
        Blob<Dtype> summer_vec_;
        Dtype num_constraints;
        int iteration;
    };

    template<typename Dtype>
    class LiftedStructSimilarityContinuousLossLayer : public LossLayer<Dtype>
    {
    public:
        explicit LiftedStructSimilarityContinuousLossLayer(const LayerParameter &param)
                : LossLayer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual inline int ExactNumBottomBlobs() const
        { return 4; }

        virtual inline const char *type() const
        { return "LiftedStructSimilarityContinuousLoss"; }

        virtual inline bool AllowForceBackward(const int bottom_index) const
        {
            return bottom_index != 1;
        }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        void loss_mine_negative(int idx, int N, int &neg_index, Dtype margin, const vector<vector<Dtype> > &sds);

        void gradient_mine_negative(int i_anchor, int N, int K, int &neg_index, Dtype loss_ij, Dtype sum_exp,
                                    const vector<vector<Dtype> > &sds, const Dtype *bin, Dtype *bout);

        Blob<Dtype> dist_sq_;  // cached for backward pass
        Blob<Dtype> dot_;
        Blob<Dtype> ones_;
        Blob<Dtype> blob_pos_diff_;
        Blob<Dtype> blob_neg_diff_;
        Blob<Dtype> loss_aug_inference_;
        Blob<Dtype> summer_vec_;
        Dtype num_constraints;
        int iteration;
    };

    /**
    * @brief Normalizes input.
    */
    template<typename Dtype>
    class NormalizeLayer : public Layer<Dtype>
    {
    public:
        explicit NormalizeLayer(const LayerParameter &param)
                : Layer<Dtype>(param)
        {}

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const
        { return "Normalize"; }

        virtual inline int ExactNumBottomBlobs() const
        { return 1; }

        virtual inline int ExactNumTopBlobs() const
        { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        Blob<Dtype> sum_multiplier_, norm_, squared_;
    };

    /**
    * @brief Get the sub-region features around some specific points
    *
    * TODO(dox): thorough documentation for Forward, Backward, and proto params.
    */
    template<typename Dtype>
    class SubRegionLayer : public Layer<Dtype>
    {
    public:
        explicit SubRegionLayer(const LayerParameter &param)
                : Layer<Dtype>(param)
        {}

        virtual void LayerSetUp(const vector<Blob<Dtype> *> &bottom,
                                const vector<Blob<Dtype> *> &top);

        virtual void Reshape(const vector<Blob<Dtype> *> &bottom,
                             const vector<Blob<Dtype> *> &top);

        virtual inline const char *type() const
        { return "SubRegion"; }

        virtual inline int ExactNumBottomBlobs() const
        { return 3; }

        virtual inline int MinTopBlobs() const
        { return 1; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        int height_;
        int width_;
        int data_height_;
        int data_width_;
        int as_dim_;
    };

    /**
    * @brief Add noise.
    */
    template<typename Dtype>
    class NoiseLayer : public NeuronLayer<Dtype>
    {
    public:
        explicit NoiseLayer(const LayerParameter &param)
                : NeuronLayer<Dtype>(param)
        {}

        virtual inline const char *type() const
        { return "Noise"; }

    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Forward_gpu(const vector<Blob<Dtype> *> &bottom,
                                 const vector<Blob<Dtype> *> &top);

        virtual void Backward_cpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

        virtual void Backward_gpu(const vector<Blob<Dtype> *> &top,
                                  const vector<bool> &propagate_down, const vector<Blob<Dtype> *> &bottom);

    };

}  // namespace caffe

#endif  // CAFFE_NEURON_LAYERS_HPP_
