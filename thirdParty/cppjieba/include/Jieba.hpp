#ifndef CPPJIEAB_JIEBA_H
#define CPPJIEAB_JIEBA_H
#include <filesystem>
#include "QuerySegment.hpp"
#include "KeywordExtractor.hpp"

namespace cppjieba {

class Jieba {
 public:
  Jieba(const string& dict_path, 
        const string& model_path,
        const string& user_dict_path, 
        const string& idfPath, 
        const string& stopWordPath) 
    : dict_trie_(dict_path, user_dict_path),
      model_(model_path),
      mp_seg_(&dict_trie_),
      hmm_seg_(&model_),
      mix_seg_(&dict_trie_, &model_),
      full_seg_(&dict_trie_),
      query_seg_(&dict_trie_, &model_),
      extractor(&dict_trie_, &model_, idfPath, stopWordPath) {
  }
  Jieba(const std::filesystem::path& cppjieba_dict) :
    Jieba(std::filesystem::path(cppjieba_dict / "jieba.dict.utf8").string(),
          std::filesystem::path(cppjieba_dict / "hmm_model.utf8").string(),
          std::filesystem::path(cppjieba_dict / "user.dict.utf8").string(),
          std::filesystem::path(cppjieba_dict / "idf.utf8").string(),
          std::filesystem::path(cppjieba_dict / "stop_words.utf8").string())
      {
          //init cppjieba
          std::filesystem::path DICT_PATH = cppjieba_dict / "jieba.dict.utf8";
          std::filesystem::path HMM_PATH = cppjieba_dict / "hmm_model.utf8";
          std::filesystem::path USER_DICT_PATH = cppjieba_dict / "user.dict.utf8";
          std::filesystem::path IDF_PATH = cppjieba_dict / "idf.utf8";
          std::filesystem::path STOP_WORD_PATH = cppjieba_dict / "stop_words.utf8";
          assert(std::filesystem::exists(DICT_PATH) && std::filesystem::exists(HMM_PATH) && std::filesystem::exists(USER_DICT_PATH) && std::filesystem::exists(IDF_PATH)
              && std::filesystem::exists(STOP_WORD_PATH) && "cppjieba dict path does not exit!");
          std::cout << "init cppjieba\n";
      }
  ~Jieba() {
  }

  struct LocWord {
    string word;
    size_t begin;
    size_t end;
  }; // struct LocWord

  void Cut(const string& sentence, std::vector<std::string>& words, bool hmm = true) const {
    mix_seg_.Cut(sentence, words, hmm);
  }
  void Cut(const string& sentence, std::vector<Word>& words, bool hmm = true) const {
    mix_seg_.Cut(sentence, words, hmm);
  }
  void CutAll(const string& sentence, std::vector<std::string>& words) const {
    full_seg_.Cut(sentence, words);
  }
  void CutAll(const string& sentence, std::vector<Word>& words) const {
    full_seg_.Cut(sentence, words);
  }
  void CutForSearch(const string& sentence, std::vector<std::string>& words, bool hmm = true) const {
    query_seg_.Cut(sentence, words, hmm);
  }
  void CutForSearch(const string& sentence, std::vector<Word>& words, bool hmm = true) const {
    query_seg_.Cut(sentence, words, hmm);
  }
  void CutHMM(const string& sentence, std::vector<std::string>& words) const {
    hmm_seg_.Cut(sentence, words);
  }
  void CutHMM(const string& sentence, std::vector<Word>& words) const {
    hmm_seg_.Cut(sentence, words);
  }
  void CutSmall(const string& sentence, std::vector<std::string>& words, size_t max_word_len) const {
    mp_seg_.Cut(sentence, words, max_word_len);
  }
  void CutSmall(const string& sentence, std::vector<Word>& words, size_t max_word_len) const {
    mp_seg_.Cut(sentence, words, max_word_len);
  }
  
  void Tag(const string& sentence, std::vector< std::pair<std::string, string> >& words) const {
    mix_seg_.Tag(sentence, words);
  }
  string LookupTag(const string &str) const {
    return mix_seg_.LookupTag(str);
  }
  bool InsertUserWord(const string& word, const string& tag = UNKNOWN_TAG) {
    return dict_trie_.InsertUserWord(word, tag);
  }

  bool InsertUserWord(const string& word,int freq, const string& tag = UNKNOWN_TAG) {
    return dict_trie_.InsertUserWord(word,freq, tag);
  }

  bool DeleteUserWord(const string& word, const string& tag = UNKNOWN_TAG) {
    return dict_trie_.DeleteUserWord(word, tag);
  }
  
  bool Find(const string& word)
  {
    return dict_trie_.Find(word);
  }

  void ResetSeparators(const string& s) {
    //TODO
    mp_seg_.ResetSeparators(s);
    hmm_seg_.ResetSeparators(s);
    mix_seg_.ResetSeparators(s);
    full_seg_.ResetSeparators(s);
    query_seg_.ResetSeparators(s);
  }

  const DictTrie* GetDictTrie() const {
    return &dict_trie_;
  } 
  
  const HMMModel* GetHMMModel() const {
    return &model_;
  }

  void LoadUserDict(const std::vector<std::string>& buf)  {
    dict_trie_.LoadUserDict(buf);
  }

  void LoadUserDict(const set<std::string>& buf)  {
    dict_trie_.LoadUserDict(buf);
  }

  void LoadUserDict(const string& path)  {
    dict_trie_.LoadUserDict(path);
  }

 private:
  DictTrie dict_trie_;
  HMMModel model_;
  
  // They share the same dict trie and model
  MPSegment mp_seg_;
  HMMSegment hmm_seg_;
  MixSegment mix_seg_;
  FullSegment full_seg_;
  QuerySegment query_seg_;

 public:
  KeywordExtractor extractor;
}; // class Jieba

} // namespace cppjieba

#endif // CPPJIEAB_JIEBA_H
