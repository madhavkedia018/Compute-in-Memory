// vgg16_cnn.cpp
// Build: use the CMakeLists provided below and link to libtorch
// Small VGG-16 variant adapted for CIFAR-10 (32x32 images)

#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <vector>

// --- VGG-like block helper ---
torch::nn::Sequential make_vgg_block(int in_ch, int out_ch, int convs = 2) {
    torch::nn::Sequential seq;
    int ch = in_ch;
    for (int i = 0; i < convs; ++i) {
        seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(ch, out_ch, 3).padding(1)));
        seq->push_back(torch::nn::BatchNorm2d(out_ch));
        seq->push_back(torch::nn::ReLU(true));
        ch = out_ch;
    }
    seq->push_back(torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(2).stride(2)));
    return seq;
}

// --- VGG-16 for CIFAR-10 (reduced channels to keep it lightweight) ---
struct VGG16Impl : torch::nn::Module {
    torch::nn::Sequential features{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};

    VGG16Impl(int num_classes = 10) {
        // Channels scaled down (original VGG uses larger filter counts)
        features = torch::nn::Sequential(
            // conv blocks: (conv-relu * N) + maxpool
            make_vgg_block(3, 64, 2),   // 32 -> 16
            make_vgg_block(64, 128, 2), // 16 -> 8
            make_vgg_block(128, 256, 3),// 8 -> 4
            make_vgg_block(256, 512, 3),// 4 -> 2
            make_vgg_block(512, 512, 3) // 2 -> 1
        );
        avgpool = torch::nn::AdaptiveAvgPool2d(torch::nn::AdaptiveAvgPool2dOptions({1, 1}));
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
        x = torch::relu(fc1->forward(x));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/this->is_training());
        x = torch::relu(fc2->forward(x));
        x = this->is_training() ? torch::dropout(x, 0.5, true) : x;
        x = fc3->forward(x);
        return x;
    }
};
TORCH_MODULE(VGG16);

// --- Training utilities ---
struct TrainOptions {
    size_t batch_size = 128;
    size_t epochs = 30;
    double lr = 0.01;
    bool use_cuda = torch::cuda::is_available();
};

void train_loop(VGG16 &model, torch::optim::Optimizer &opt, torch::data::DataLoader<torch::data::datasets::CIFAR10> &loader, TrainOptions opts, size_t epoch) {
    model->train();
    size_t batch_idx = 0;
    double running_loss = 0;
    size_t correct = 0;
    size_t total = 0;
    for (auto &batch : loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();

        if (opts.use_cuda) {
            data = data.to(torch::kCUDA);
            targets = targets.to(torch::kCUDA);
        }

        opt.zero_grad();
        auto output = model->forward(data);
        auto loss = torch::nn::functional::cross_entropy(output, targets);
        loss.backward();
        opt.step();

        running_loss += loss.template item<double>() * data.size(0);
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
        total += data.size(0);

        if (++batch_idx % 50 == 0) {
            std::cout << "Epoch: " << epoch << " Batch: " << batch_idx << " Loss: " << loss.template item<double>() << "\n";
        }
    }
    std::cout << "Epoch " << epoch << " Train loss: " << running_loss/total << " Acc: " << (double)correct/total << "\n";
}

void test_loop(VGG16 &model, torch::data::DataLoader<torch::data::datasets::CIFAR10> &loader, TrainOptions opts) {
    model->eval();
    torch::NoGradGuard no_grad;
    size_t correct = 0, total = 0;
    double loss_sum = 0;
    for (auto &batch : loader) {
        auto data = batch.data;
        auto targets = batch.target.squeeze();
        if (opts.use_cuda) {
            data = data.to(torch::kCUDA);
            targets = targets.to(torch::kCUDA);
        }
        auto output = model->forward(data);
        loss_sum += torch::nn::functional::cross_entropy(output, targets, torch::nn::functional::CrossEntropyFuncOptions().reduction(torch::kSum)).template item<double>();
        auto pred = output.argmax(1);
        correct += pred.eq(targets).sum().template item<int64_t>();
        total += data.size(0);
    }
    std::cout << "Test loss: " << loss_sum/total << " Test acc: " << (double)correct/total << "\n";
}

int main() {
    TrainOptions opts;
    std::cout << "CUDA available: " << opts.use_cuda << "\n";
    auto device = opts.use_cuda ? torch::kCUDA : torch::kCPU;

    // Datasets
    const std::string cifar_root = "./data";
    auto train_dataset = torch::data::datasets::CIFAR10(cifar_root)
        .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*std=*/{0.2470, 0.2435, 0.2616}))
        .map(torch::data::transforms::Stack<>());
    auto test_dataset = torch::data::datasets::CIFAR10(cifar_root, torch::data::datasets::CIFAR10::Mode::kTest)
        .map(torch::data::transforms::Normalize<>(/*mean=*/{0.4914, 0.4822, 0.4465}, /*std=*/{0.2470, 0.2435, 0.2616}))
        .map(torch::data::transforms::Stack<>());

    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
        std::move(train_dataset), torch::data::DataLoaderOptions().batch_size(opts.batch_size).workers(2));

    auto test_loader = torch::data::make_data_loader<torch::data::datasets::CIFAR10>(
        std::move(test_dataset), torch::data::DataLoaderOptions().batch_size(256).workers(2));

    // Model
    VGG16 model;
    model->to(device);

    torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(opts.lr).momentum(0.9).weight_decay(5e-4));

    for (size_t epoch = 1; epoch <= opts.epochs; ++epoch) {
        train_loop(model, optimizer, *train_loader, opts, epoch);
        test_loop(model, *test_loader, opts);
        // LR schedule simple step
        if (epoch == 20) {
            for (auto &g : optimizer.param_groups()) {
                static_cast<torch::optim::SGDOptions&>(g.options()).lr( opts.lr * 0.1 );
            }
            std::cout << "LR decayed\n";
        }
    }

    // Save final model
    torch::save(model, "vgg16_cifar10.pt");
    std::cout << "Saved model to vgg16_cifar10.pt\n";
    return 0;
}
