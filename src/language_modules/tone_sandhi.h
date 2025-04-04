/**
 * Copyright      2024    Tong Qiu (tong.qiu@intel.com)  Haofan Rong (haofan.rong@hp.com)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once
#ifndef TONE_SANDHI_H
#define TONE_SANDHI_H
#include <string>
#include <utility>
#include <vector>

#include "Jieba.hpp"
#include "tokenizer.h"
namespace melo {
namespace ToneSandhi {
std::vector<std::pair<std::string, std::string>> pre_merge_for_modify(
    std::vector<std::pair<std::string, std::string>>& seg);
/*    void _merge_continuous_three_tones(std::vector<std::pair<std::string, std::string>>& seg);
    void _merge_continuous_three_tones_2(std::vector<std::pair<std::string, std::string>>&seg);*/
std::vector<std::pair<std::string, std::string>> _merge_yi(std::vector<std::pair<std::string, std::string>>& seg);
std::vector<std::pair<std::string, std::string>> _merge_chinese_patterns(
    std::vector<std::pair<std::string, std::string>>& seg);

void modified_tone(const std::string& word,
                   const std::string& tag,
                   const std::shared_ptr<cppjieba::Jieba>& jieba,
                   std::vector<std::string>& sub_finals);  // input is word by word
void _bu_sandhi(const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
void _yi_sandhi(const std::vector<std::string>& chinese_characters, std::vector<std::string>& sub_finals);
void _neural_sandhi(const std::string& word,
                    const std::vector<std::string>& chinese_characters,
                    const std::string& tag,
                    const std::shared_ptr<cppjieba::Jieba>& jieba,
                    std::vector<std::string>& sub_finals);
void _three_sandhi(const std::string& word,
                   const std::vector<std::string>& chinese_characters,
                   const std::shared_ptr<cppjieba::Jieba>& jieba,
                   std::vector<std::string>& sub_finals);

const static std::set<std::string> numeric =
    {"零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百", "千", "万", "亿", "兆"};

inline bool is_numeric(const std::string& chinese_character) {
    return numeric.contains(chinese_character);
}
inline bool is_numeric(const std::vector<std::string>& chinese_characters) {
    return std::all_of(chinese_characters.begin(), chinese_characters.end(), [](const std::string& ch) {
        return is_numeric(ch);
    });
}

}  // namespace ToneSandhi
const static std::unordered_set<std::string> must_not_neural_tone_words =
    {"男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子", "电子", "人人", "虎虎"};
const static std::unordered_set<std::string> must_neural_tone_words = {
    "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰", "馒头", "馄饨", "风筝", "难为", "队伍", "阔气",
    "闺女", "门道", "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊", "里头", "部分", "那么", "道士", "造化", "迷糊",
    "连累", "这么", "这个", "运气", "过去", "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主", "豆腐", "讲究",
    "记性", "记号", "认识", "规矩", "见识", "裁缝", "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆",
    "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头", "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在",
    "膏药", "脾气", "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同", "聪明", "耽误", "耽搁", "耷拉",
    "耳朵", "老爷", "老实", "老婆", "老头", "老太", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火", "累赘", "糨糊",
    "糊涂", "精神", "粮食", "簸箕", "篱笆", "算计", "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿", "窝囊", "窗户",
    "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗", "砚台", "码头", "石榴", "石头", "石匠", "知识",
    "眼睛", "眯缝", "眨巴", "眉毛", "相声", "盘算", "白净", "痢疾", "痛快", "疟疾", "疙瘩", "疏忽", "畜生", "生意",
    "甘蔗", "琵琶", "琢磨", "琉璃", "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜", "牌楼", "爽快",
    "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心", "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚",
    "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔", "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火",
    "架势", "枕头", "枇杷", "机灵", "本事", "木头", "木匠", "朋友", "月饼", "月亮", "暖和", "明白", "时候", "新鲜",
    "故事", "收拾", "收成", "提防", "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌", "招呼", "抬举",
    "护士", "折腾", "扫帚", "打量", "打算", "打点", "打扮", "打听", "打发", "扎实", "扁担", "戒指", "懒得", "意识",
    "意思", "情形", "悟性", "怪物", "思量", "怎么", "念头", "念叨", "快活", "忙活", "志气", "心思", "得罪", "张罗",
    "弟兄", "开通", "应酬", "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结", "巴掌", "差事", "工夫",
    "岁数", "屁股", "尾巴", "少爷", "小气", "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在", "官司",
    "学问", "学生", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家", "委屈", "姑娘", "姐夫", "妯娌", "妥当", "妖精",
    "奴才", "女婿", "头发", "太阳", "大爷", "大方", "大意", "大夫", "多少", "多么", "外甥", "壮实", "地道", "地方",
    "在乎", "困难", "嘴巴", "嘱咐", "嘟囔", "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴", "哈欠", "哆嗦",
    "咳嗽", "和尚", "告诉", "告示", "含糊", "吓唬", "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋", "厚道",
    "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹", "功夫", "力气", "前头", "刺猬", "刺激", "别扭",
    "利落", "利索", "利害", "分析", "出息", "凑合", "凉快", "冷战", "冤枉", "冒失", "养活", "关系", "先生", "兄弟",
    "便宜", "使唤", "佩服", "作坊", "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚", "亲家", "交情",
    "云彩", "事情", "买卖", "主意", "丫头", "丧气", "两口", "东西", "东家", "世故", "不由", "不在", "下水", "下巴",
    "上头", "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲", "咕噜", "邋遢", "费用", "冤家", "甜头",
    "介绍", "荒唐", "大人", "泥鳅", "幸福", "熟悉", "计划", "扑腾", "蜡烛", "姥爷", "照顾", "喉咙", "吉他", "弄堂",
    "蚂蚱", "凤凰", "拖沓", "寒碜", "糟蹋", "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把", "惦记",
};
inline size_t _split_word(const std::string& word, const std::shared_ptr<cppjieba::Jieba>& jieba) {
    std::vector<cppjieba::Word> words;
    jieba->CutForSearch(word, words);
    cppjieba::Word wordForSearch = words[0];
    if (wordForSearch.unicode_offset == 0) {
        if (wordForSearch.unicode_length != word.size()) {
            return wordForSearch.unicode_length;
        }
    } else {
        return wordForSearch.unicode_offset;
    }
    return 0;
}
}  // namespace melo
#endif  // TONE_SANDHI_H