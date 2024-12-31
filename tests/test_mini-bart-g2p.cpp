#include <cassert>
#include "mini-bart-g2p.h"
using namespace melo;
#define OV_MODEL_PATH "ov_models"

// set ONEDNN_CACHE_CAPACITY 
void ConfigureOneDNNCache() {
#ifdef _WIN32
    auto status = _putenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY=100");
#elif __linux__ 
    auto status = setenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY", "100", true);
#else
    std::cout << "Running on an unknown OS" << std::endl;
#endif
    // TODO : Add try catch block here
    if (status == 0) {
        char* onednn_kernel_capacity = std::getenv("ONEDNN_PRIMITIVE_CACHE_CAPACITY");
        int num = std::stoi(std::string(onednn_kernel_capacity));
        assert((num == 100) && "[ERROR] Set ONEDNN_PRIMITIVE_CACHE_CAPACITY fails!");
        std::cout << "set ONEDNN_PRIMITIVE_CACHE_CAPACITY: " << onednn_kernel_capacity << "\n";
    }
}


int main(){
    ConfigureOneDNNCache();
	std::filesystem::path  model_folder = "C:\\Users\\gta\\source\\repos\\MeloTTS.cpp\\experimental\\mini-bart-g2p\\mini-bart-g2p-no_cache";
	bool use_past = false;
	std::unique_ptr<ov::Core> core = std::make_unique<ov::Core>();
	MiniBartG2P g2p(core,model_folder,"CPU", use_past);
	g2p.forward("hello");//s</s> <s> HH EH1 L OW0 </s>
	//system("pause");
}