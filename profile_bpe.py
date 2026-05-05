import cProfile, pstats                                                                                 
from pathlib import Path
from cs336_basics.bpe import train_bpe                                                                  
                                                                                                        
FIXTURES = Path('tests/fixtures')
                                                                                                        
prof = cProfile.Profile()                                       
prof.enable()                                                                                           
train_bpe(
    input_path=FIXTURES / 'corpus.en',                                                                  
    vocab_size=500,                                             
    special_tokens=['<|endoftext|>'],                                                                   
)
prof.disable()                                                                                          
                                                                
stats = pstats.Stats(prof).sort_stats('cumulative')
stats.print_stats(25)                                                                                   
print('--- by tottime ---')
pstats.Stats(prof).sort_stats('tottime').print_stats(25) 