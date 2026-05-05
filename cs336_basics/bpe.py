from collections import defaultdict
from collections import Counter
import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""                                                                        

def train_bpe(input_path, vocab_size, special_tokens):                                                                                          
    # serial init                       
    with open(input_path, encoding="utf-8") as f:
        train_material = f.read()

    special_tokens_sort = sorted(special_tokens, key=len, reverse=True)
    if special_tokens:
        reg_special = "|".join(re.escape(t) for t in special_tokens_sort)
        parts = re.split(reg_special, train_material)                                                        
    else:
        parts = [train_material]

    word_freq = Counter()
    for part in parts:
        words = re.findall(PAT, part)                                                                         
        for word in words:
            old_word = tuple(bytes([b]) for b in word.encode("utf-8"))
            word_freq[old_word] += 1
                                                                                                            
                                                                                                            
    # vocab初始化（初始 vocab、初始序列）                                                                  
    vocab = {i:bytes([i]) for i in range(256)}                                                              
    for i in range(len(special_tokens)):                                                                    
        vocab[256+i] = special_tokens[i].encode("utf-8")                                                        
                                                                                                            
    merges = []               
    
    # 3. 主循环（统计、tie-breaking、合并、更新序列、记录 merge）                                 
    count = defaultdict(int)                                                                                          
    pair_words = defaultdict(set)    # key: tuple of bytes, val: wordA, wordB       
    # 统计                                                                                             
    for old_word,freq in word_freq.items():                                                                           
        for i in range(len(old_word)-1):                                                                    
            this_token = (old_word[i], old_word[i+1])                                                      
            count[this_token]+= freq # count is dict       
            pair_words[this_token].add(old_word)                                                                                                               
    
    while len(vocab) < vocab_size:                                                                        
        # tie-breaking                                                                                     
        max_freq = max(count.values())                                                                      
        candidates = {k for (k,v) in count.items() if v==max_freq} #/ sort by val, return key. maybe several 
        tar_token = max(candidates)                                                                         
                                                                                                            
        # 合并                                                                                             
        new_token = tar_token[0] + tar_token[1] # two token merge into one new token       

        # 更新序列                                                                                         
        affected_words = list(pair_words[tar_token])   
        for old_word in affected_words:  
            freq = word_freq[old_word]
            
            # --- 第一步：抹除旧痕迹 (非常关键！) ---
            # 既然 old_word 要变了，它之前贡献的所有 pair 频率都要从 count 里减去
            # 并且它也要从 pair_words 的那些集合里把自己删掉
            for j in range(len(old_word)-1):
                p = (old_word[j], old_word[j+1])
                count[p] -= freq
                pair_words[p].discard(old_word)
            
            # --- 第二步：生成新词 (你已经写了一部分) ---
            new_word = []                                                                                   
            i = 0                                                                                             
            while i < len(old_word):                                                                            
                if i+1 < len(old_word) and (old_word[i], old_word[i+1]) == tar_token:                                 
                    new_word.append(new_token)                                                                
                    i += 2           # 跳过下一个，避免重叠                                                      
                else:                                                                                           
                    new_word.append(old_word[i])                                                                
                    i += 1                                                                                 
            # ... (你的替换逻辑) ...
            new_word = tuple(new_word)
            
            # --- 第三步：登记新痕迹 ---
            # 把新词 new_word 产生的新 pair 频率加上去，并更新索引
            for j in range(len(new_word)-1):
                p = (new_word[j], new_word[j+1])
                count[p] += freq
                pair_words[p].add(new_word)
            
            # --- 第四步：同步 word_freq ---
            # 注意：这里千万不要 word_freq = new_word_freq，那是全量替换
            # 应该是只替换这一个词
            del word_freq[old_word]
            word_freq[new_word] += freq                                                                                  
                                                                                                            
        vocab[len(vocab)] = new_token                                                                       
                                                                                                            
        # 记录 merge                                                                                       
        merges.append(tar_token) 

    return vocab, merges