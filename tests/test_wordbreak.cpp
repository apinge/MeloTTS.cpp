#define CRT_
#ifdef CRT_
#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>
#endif
#include <iostream>
#include <cassert>
#include <filesystem>
#include <functional>
#include <queue>
#include "cmudict.h"

#define OV_MODEL_PATH "ov_models"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::milliseconds ms;
typedef std::chrono::microseconds us;
inline long long get_duration_ms_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<ms>(Time::now() - startTime).count();;
};
inline long long get_duration_us_till_now(Time::time_point& startTime) {
    return std::chrono::duration_cast<us>(Time::now() - startTime).count();;
};


auto print_result = [](const auto& result) {
    if (result.has_value()) {
        const auto& vec = result.value().get();  // get the ref
        for (const auto& inner_vec : vec) {
            for (const auto& str : inner_vec) {
                std::cout << str << ".";
            }
            std::cout << ",";
        }
    }
    else {
        std::cout << "Key not found" << std::endl;
    }
    std::cout << std::endl;
    };

std::vector<std::string> wordBreak1(const std::string& s, const std::unordered_map<std::string, std::vector<std::vector<std::string>>>& st ) {
    int n = s.length();
    std::vector<std::string> res;
    std::vector<std::vector<std::pair<std::string, int>>> dp(n);//{自己，前一个位置}
    for (int i = 0; i < n; ++i) {
        //wordDict 数据长度很小
        for (const auto& [word,_] : st) {
            int length = word.length();
            if (i + 1 < length) continue;
            auto cur = s.substr(i - length + 1, length);
            if (cur != word) continue;
            //确认i是否为word的最后一个字母
            if (i == length - 1)
                dp[i].emplace_back(word, -1);//dp[0,i]为一个词
            else if (dp[i - length].size()) {
                dp[i].emplace_back(word, i - length);
            }
        }

    }
    if (!dp.back().size()) return {};
    std::queue<std::pair<std::string, int>> q;
    for (const auto& [s, prevId] : dp.back()) {
        if (prevId == -1)
            res.emplace_back(s);
        else
            q.emplace(s, prevId);
    }
    while (!q.empty()) {
        auto [str, prevId] = q.front(); q.pop();

        for (const auto& [prevStr, prevprevId] : dp[prevId]) {
            std::string cur = prevStr + ' ' + str;
            if (prevprevId == -1)
                res.emplace_back(cur);
            else
                q.emplace(cur, prevprevId);
        }
    }
    return res;
}

std::vector<std::string> wordBreak2(const std::string& s, const std::unordered_map<std::string, std::vector<std::vector<std::string>>>& word) {
    int n = s.size();
   
    int mask = 1 << (n - 1);//添加空格情况总数

    std::vector<std::string> ans;//答案;

    for (int a = 0; a < mask; a++)
    {
        std::string sb;
        bool ok = true;
        for (int i = 0, j = -1; i < n && ok; i++)
        {
            sb.push_back(s[i]);

            if (((a >> i) & 1) == 1 || i == n - 1)
            {

                if (!(word.find(sb.substr(j + 1, sb.length())) != word.end())) ok = false;//如果找不到整个单词
                if (i != n - 1) sb.push_back(' ');//不是最后一个单词
                j = sb.length() - 1;//当前添加单词的末尾 

            }
        }
        //cout << ok << endl;
        if (ok) ans.push_back(sb);
    }
    return ans;
}

std::vector<std::string> wordBreak(const std::string& s, const std::unordered_map<std::string, std::vector<std::vector<std::string>>> dict_) {
    std::unordered_map<int, std::vector<std::string>> ans;
    std::function<void(const std::string& s, int index)> backtrack = [&](const std::string& s, int index) {
        if (!ans.contains(index)) {
            if (index == s.size()) {
                ans[index] = { "" };
                return;
            }
            ans[index] = {};
            // assume string.size() >=2, because single-character words were excluded to reduce the number of permutations.
            for (int i = index + 2; i <= s.size(); ++i) {
                std::string word = s.substr(index, i - index);
                if (dict_.contains(word)) {
                    backtrack(s, i);
                    for (const std::string& succ : ans[i]) {
                        ans[index].push_back(succ.empty() ? word : word + " " + succ);
                    }
                }
            }
        }
    };

    backtrack(s, 0);
    return ans[0];
}

std::vector<std::string> wordBreak_da(const std::string& s, melo::CMUDict& dict) {
    std::unordered_map<int, std::vector<std::string>> ans;
    std::function<void(const std::string& s, int index)> backtrack = [&](const std::string& s, int index) {
        if (!ans.contains(index)) {
            if (index == s.size()) {
                ans[index] = { "" };
                return;
            }
            ans[index] = {};
            // assume string.size() >=2, because single-character words were excluded to reduce the number of permutations.
            for (int i = index + 2; i <= s.size(); ++i) {
                const std::string word = s.substr(index, i - index);
                if (dict.contains(word)) {
                    backtrack(s, i);
                    for (const std::string& succ : ans[i]) {
                        ans[index].push_back(succ.empty() ? word : word + " " + succ);
                    }
                }
            }
        }
        };

    backtrack(s, 0);
    return ans[0];
}



int main() {
    {
        std::string ov_models = "C:\\Users\\gta\\source\\repos\\MeloTTS.cpp\\ov_models";
        std::filesystem::path file_dir = std::filesystem::path(ov_models) / "cmudict_cache_order.txt";
        //std::cout << std::boolalpha << std::filesystem::absolute(file_dir) << std::endl;
        melo::CMUDict dict(file_dir);

        //std::unordered_map<std::string, std::vector<std::vector<std::string>>> mp;



        //plan search
        const std::vector<std::string>& res1 = dict.direct_lookup("compiler");
        for (std::cout << "find word compiler:"; auto & row : res1) for (auto& x : row) std::cout << x << ' ';

        std::string text = "deletepartitionoverride";//"masterbootrecordsecurity";
        auto startTime = Time::now();
        auto res = dict.wordBreak(text);
        auto execTime = get_duration_ms_till_now(startTime);
        std::cout << "execTime " << execTime << "ms\n";
        std::cout << "found " << res.size() << "answer" << std::endl;
        if (res.size()) for (auto& row : res) {
            for (auto& x : row) std::cout << x << ' ';
            std::cout << std::endl;
        }
        auto symbols = dict.find(text);
        if (symbols.has_value()) {
            for (auto& row : symbols.value()) {
                for (auto& x : row)
                    std::cout << x << ' ';
                std::cout << std::endl;
            }
        }
    }
    /*
    * windowsdefendersmartscreen
    * win do ws de fend er smart screen
    win do ws de fend ers mart screen
    win do ws de fender smart screen
    win do ws de fenders mart screen
    win do ws defend er smart screen
    win do ws defend ers mart screen
    win do ws defender smart screen
    win do ws defenders mart screen
    windows de fend er smart screen
    windows de fend ers mart screen
    windows de fender smart screen
    windows de fenders mart screen
    windows defend er smart screen
    windows defend ers mart screen
    windows defender smart screen
    windows defenders mart screen
    downloadcloudrecoveryclient
    down lo ad cloud re co very client
    down lo ad cloud rec overy client
    down lo ad cloud reco very client
    down lo ad cloud recovery client
    down load cloud re co very client
    down load cloud rec overy client
    down load cloud reco very client
    down load cloud recovery client
    download cloud re co very client
    download cloud rec overy client
    download cloud reco very client
    download cloud recovery client


    */
   

#ifdef CRT_
    _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_DEBUG);
    _CrtDumpMemoryLeaks();
#endif
}