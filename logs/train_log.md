# Q



* What about weights and backpropagation?



* What does it mean that the decoder focus[es]" on different parts of the source sentence as it generates each word"? How does this relate to the size of the context window?
* 





without beam search:

2025-03-16 14:24:59 - INFO - BLEU score: 0.21

2025-03-16 14:24:59 - INFO - chrF score: 8.7996

without beacm search, fixed:

2025-03-16 14:27:34 - INFO - BLEU score: 0.46

2025-03-16 14:27:34 - INFO - chrF score: 9.0228

with beam search:

2025-03-16 14:39:04 - INFO - BLEU score: 0.80
2025-03-16 14:39:04 - INFO - chrF score: 13.957

with beam search, fixed:
2025-03-16 14:29:32 - INFO - BLEU score: 1.46
2025-03-16 14:29:32 - INFO - chrF score: 14.9398