#include <iostream>
#include "Datasets.cuh"

int main()
{
    try
    {
        // --- Create and configure the loader ---
        OpenCVLoader loader;
        loader.batch_size = 32;
        loader.target_channels = 3;
        loader.target_height = 64;
        loader.target_width = 64;
        loader.cudaEnabled = false; // or true if CUDA is available

        // --- Load datasets ---
        loader.load_from_folders();
        std::map<std::string, int> class_map = loader.local_class_to_label;

        // --- Summary statistics ---
        std::cout << "\n==== Load Summary ====\n";
        std::cout << "Classes found: \n";
        for (const auto &pair : class_map)
        {
            std::cout << "  " << pair.first << " => " << pair.second << "\n";
        }

        std::cout << "Training batches: " << loader.training_batches.size() << "\n";
        std::cout << "Testing batches:  " << loader.testing_batches.size() << "\n";
        std::cout << "Validation batches: " << loader.validation_batches.size() << "\n";

        if (!loader.training_batches.empty())
        {
            std::cout << "\nExample training batch shape: ";
            loader.training_batches[0].print_shape();

            std::cout << "Example label batch shape: ";
            loader.training_label_batches[0].print_shape();

            std::cout << "Sample label values in first batch: ";
            for (size_t i = 0; i < std::min((size_t)10, loader.training_label_batches[0].get_shape()[0]); ++i)
                std::cout << loader.training_label_batches[0](i, 0) << " ";
            std::cout << "\n";
        }

        if (loader.cudaEnabled && !loader.training_batches_cuda.empty())
        {
            std::cout << "\n[CUDA] Training batch shape: ";
            const auto &shape = loader.training_batches_cuda[0].get_shape();
            std::cout << "[" << shape[0] << ", " << shape[1] << ", " << shape[2] << ", " << shape[3] << "]\n";
        }
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Exception occurred during data loading: " << ex.what() << std::endl;
        return 1;
    }

    return 0;
}
