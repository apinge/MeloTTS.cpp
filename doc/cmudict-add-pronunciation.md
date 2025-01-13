# [MeloTTS.cpp] A Guide to Working with the Custom CMUDict Data Table
## CMU Pronouncing Dictionary
[CMU Pronouncing Dictionary](https://en.wikipedia.org/wiki/CMU_Pronouncing_Dictionary) is an open-source pronouncing dictionary that provides orthographic-to-phonetic mappings for English words. In this repo, we maintain a local, customized CMUDict based on the original CMUDict, with some additions tailored to our specific needs. 
## Data Structure and Implementation of the Custom CMUDict Table
In this repo, [cmudict.cpp](../src/language_modules/cmudict.cpp) and [cmudict.h](../src/language_modules/cmudict.h) implement the dictionary lookup functionality. Unit tests for this functionality are provided in [test_wordbreak.cpp](../tests/test_wordbreak.cpp). Dictionary data for online use is stored in [cmudict_cache_order.txt](../ov_models/cmudict_cache_order.txt), while the raw data resides in [cmudict_cache.txt](../scripts/cmudict_cache.txt). **Be aware that data files can vary across different branches.**

### Support Compound Word
We've added some logic to handle compound words like "windowspowershell." Basically, it splits the word into parts (like "windows" and "powershell"), looks up the pronunciation for each part in CMUDict, and then puts them together. We use memoized dfs (see `CMUDict::wordBreak` in [cmudict.cpp](../src/language_modules/cmudict.cpp)) to make the splitting faster, and a [double-array trie](../thirdParty/cppinyin/csrc/darts.h) to make the dictionary lookups super quick.

Despite the significant performance gains offered by these algorithms, The process of adding new words to CMUDict is somewhat complex. 

### Add new words to Custom CMUDict

* First, ensure that the keys you intend to add are unique within [cmudict_cache.txt](../scripts/cmudict_cache.txt). If a key already exists, modify its corresponding value directly.
* Execute the  [cmudict_cache.py](../scripts/cmudict_cache.py) script. This will process [cmudict_cache.txt](../scripts/cmudict_cache.txt), handling any words containing special characters and sorting the keys lexicographically (alphabetically) to ensure compatibility with the double-array trie data structure. The generated output, `cmudict_cache_order.txt`
* Place the new `cmudict_cache_order.txt` in folder [ov_models](../ov_models/), replacing the existing `cmudict_cache_order.txt` file.

Because word breaking is supported, adding compound words is simplified to adding their constituent words. For example, to add "EdgeCloudSync," and given that "edge" and "cloud" are already in the CMUDict dictionary, only "sync" needs to be added.