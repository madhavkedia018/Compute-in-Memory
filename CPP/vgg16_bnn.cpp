// vgg16_bnn.cpp
// Binarized VGG-like net for CIFAR-10 using libtorch
// Note: BNN training is sensitive — treat this as a starting point.

#include <torch/torch.h>
#include <iostream>

// Straight-Through Estimator helpers
torch::Tensor binarize_activation(const torch::Tensor &x) {
    // forward: sign(x) where sign(0)=1; use STE: return sign(x) in forward, but identity gradient
    auto sign_x = x.sign();
    // ensure no zeros: sign(0) = 1 (optional)
    sign_x = torch::where(sign_x.eq(0), torch::ones_like(sign_x), sign_x);
    return (sign_x - x).detach() + x; // STE: gradient flows through x
}

torch::Tensor binarize_weight_with_scaling(const torch::Tensor &w) {
    // w: [out_ch, in_ch, kH, kW]
    // compute per-filter scaling alpha = mean(abs(w))
    auto w_view = w.view({w.size(0), -1});
    auto alpha = w_view.abs().mean(1, /*keepdim=*/true);
    alpha = alpha.view({-1, 1, 1, 1}); // shape: [out_ch,1,1,1]
    auto w_sign = w.sign();
    w_sign = torch::where(w_sign.eq(0), torch::ones_like(w_sign), w_sign);
    auto w_bin = (w_sign * alpha).detach() + (w - w).detach(); // forward uses w_sign*alpha, gradient flows to w through STE pattern
    // The expression above uses detach to freeze the forward value but keeps gradient path; simpler:
    // return (w_sign * alpha - w).detach() + w; // equivalent
    return (w_sign * alpha - w).detach() + w;
}

// Binarized convolutional layer
struct BinarizedConv2dImpl : torch::nn::Module {
    torch::nn::Conv2d conv{nullptr};
    bool bias;

    BinarizedConv2dImpl(int in_ch, int out_ch, int k=3, int stride=1, int padding=1, bool with_bias=false) {
        conv = register_module("conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(in_ch, out_ch, k).stride(stride).padding(padding).bias(with_bias)));
        bias = with_bias;
    }

    torch::Tensor forward(torch::Tensor x) {
        // BatchNorm and ReLU are usually kept as float (not binarized) in many BNN variants.
        // Binarize input activation
        auto x_bin = binarize_activation(x);
        // Binarize weights with per-filter scaling
        auto w = conv->weight;
        auto w_bin = binarize_weight_with_scaling(w);
        // perform conv using binarized weights
        return torch::conv2d(x_bin, w_bin, conv->bias, conv->options.stride(), conv->options.padding(), conv->options.dilation(), conv->options.groups());
    }
};
TORCH_MODULE(BinarizedConv2d);

// Similar to standard VGG but using BinarizedConv2d
BinarizedConv2d make_bbin_block(int in_ch, int out_ch, int convs = 2) {
    // returns a sequential with bin convs, batchnorm, relu, pool
    // but helper cannot return sequential easily — we'll build sequential in constructor below.
    // left unused
    return nullptr;
}

struct BNNVGGImpl : torch::nn::Module {
    torch::nn::Sequential features{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    BNNVGGImpl(int num_classes = 10) {
        features = torch::nn::Sequential();
        // block 1
        features->push_back(BinarizedConv2d(3, 64, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(64));
        features->push_back(torch::nn::Hardtanh(-1, 1)); // often BNN uses hardtanh before binarization
        features->push_back(BinarizedConv2d(64, 64, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(64));
        features->push_back(torch::nn::MaxPool2d(2));

        // block 2
        features->push_back(BinarizedConv2d(64, 128, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(128));
        features->push_back(torch::nn::Hardtanh(-1,1));
        features->push_back(BinarizedConv2d(128, 128, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(128));
        features->push_back(torch::nn::MaxPool2d(2));

        // block 3
        features->push_back(BinarizedConv2d(128, 256, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(256));
        features->push_back(torch::nn::Hardtanh(-1,1));
        features->push_back(BinarizedConv2d(256, 256, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(256));
        features->push_back(BinarizedConv2d(256, 256, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(256));
        features->push_back(torch::nn::MaxPool2d(2));

        // block 4
        features->push_back(BinarizedConv2d(256, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(torch::nn::Hardtanh(-1,1));
        features->push_back(BinarizedConv2d(512, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(BinarizedConv2d(512, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(torch::nn::MaxPool2d(2));

        // block 5
        features->push_back(BinarizedConv2d(512, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(torch::nn::Hardtanh(-1,1));
        features->push_back(BinarizedConv2d(512, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(BinarizedConv2d(512, 512, 3, 1, 1));
        features->push_back(torch::nn::BatchNorm2d(512));
        features->push_back(torch::nn::MaxPool2d(2));

        avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1,1}));
        fc1 = register_module("fc1", torch::nn::Linear(512, 512));
        fc2 = register_module("fc2", torch::nn::Linear(512, 256));
        fc3 = register_module("fc3", torch::nn::Linear(256, num_classes));

        register_module("features", features);
        register_module("avgpool", avgpool);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = features->forward(x);
        x = avgpool->forward(x);
        x = x.view({x.size(0), -1});
        // Keep FC layers in float (not binarized) commonly
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, 0.5, this->is_training());
        x = torch::relu(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(BNNVGG);

// Training loops are identical to the CNN example. For brevity, you can reuse the train/test loops from vgg16_cnn.cpp
// with model type changed to BNNVGG.

int main() {
    std::cout << "BNN VGG example start\n";
    bool use_cuda = torch::cuda::is_available();
    auto device = use_cuda ? torch::kCUDA : torch::kCPU;
    BNNVGG model;
    model->to(device);

    // Dataset setup (same normalization)
    const std::string cifar_root = "./data";
    auto train_dataset = torch::data::datasets::CIFAR10(cifar_root)
        .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*std=*/{0.2470, 0.2435, 0.2616}))
        .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::CIFAR10(cifar_root, torch::data::datasets::CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*std=*/{0.2470, 0.2435, 0.2616}))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader(std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(128).workers(2));
    auto test_loader = torch::data::make_data_loader(std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(256).workers(2));

    // Optimizer
    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(0.02).momentum(0.9).weight_decay(5e-4));

    size_t epochs = 60;
    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        // training loop (simple)
        model->train();
        size_t total = 0, correct = 0;
        double loss_accum = 0;
        for (auto &batch : *train_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.squeeze().to(device);

            optimizer.zero_grad();
            auto output = model->forward(data);
            auto loss = torch::nn::functional::cross_entropy(output, target);
            loss.backward();
            optimizer.step();

            loss_accum += loss.template item<double>() * data.size(0);
            auto pred = output.argmax(1);
            correct += pred.eq(target).sum().template item<int64_t>();
            total += data.size(0);
        }
        std::cout << "Epoch " << epoch << " Train loss: " << loss_accum/total << " Acc: " << (double)correct/total << "\n";

        // test
        model->eval();
        torch::NoGradGuard no_grad;
        size_t tcorrect = 0, ttotal = 0;
        double test_loss = 0;
        for (auto &batch : *test_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.squeeze().to(device);
            auto out = model->forward(data);
            test_loss += torch::nn::functional::cross_entropy(out, target, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)).template item<double>();
            tcorrect += out.argmax(1).eq(target).sum().template item<int64_t>();
            ttotal += data.size(0);
        }
        std::cout << "Test loss: " << test_loss/ttotal << " Test acc: " << (double)tcorrect/ttotal << "\n";
    }

    torch::save(model, "bnn_vgg_cifar10.pt");
    std::cout << "Saved BNN model\n";
    return 0;
}
