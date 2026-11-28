"""Hinglish versions of the 250 held-out real-document questions (English -> Hinglish)."""

HINGLISH_2 = {
    "In the 'Attention Is All You Need' paper, what is the inner-layer dimensionality d_ff of the position-wise feed-forward network in the Transformer?":
        "'Attention Is All You Need' paper me Transformer ke position-wise feed-forward network ki inner-layer dimensionality d_ff kitni hai?",
    "In the 'Attention Is All You Need' paper, what values of beta1 and beta2 are used for the Adam optimizer?":
        "'Attention Is All You Need' paper me Adam optimizer ke liye beta1 aur beta2 ki kaun si values use hui hain?",
    "In the 'Attention Is All You Need' paper, what label smoothing value (epsilon_ls) is used during training?":
        "'Attention Is All You Need' paper me training ke dauran label smoothing (epsilon_ls) ki value kya hai?",
    "About how many sentence pairs are in the WMT 2014 English-German training dataset used in the 'Attention Is All You Need' paper?":
        "'Attention Is All You Need' paper me use hue WMT 2014 English-German training dataset me lagbhag kitne sentence pairs hain?",
    "For how many training steps were the big Transformer models trained in the 'Attention Is All You Need' paper?":
        "'Attention Is All You Need' paper me big Transformer models kitne training steps tak train hue the?",
    "In the 'Attention Is All You Need' paper, by how many BLEU points is single-head attention worse than the best setting in the model-variation experiments?":
        "'Attention Is All You Need' paper ke model-variation experiments me single-head attention best setting se kitne BLEU points kharab hai?",
    "In the 'Attention Is All You Need' paper, what beam size and length penalty alpha are used at inference for English constituency parsing?":
        "'Attention Is All You Need' paper me English constituency parsing ke inference me beam size aur length penalty alpha kya hain?",
    "In Table 2 of the 'Attention Is All You Need' paper, what English-to-French BLEU score does the ConvS2S Ensemble get?":
        "'Attention Is All You Need' paper ki Table 2 me ConvS2S Ensemble ko English-to-French me kitna BLEU score milta hai?",
    "In Table 4 of the 'Attention Is All You Need' paper, what WSJ 23 F1 does the 4-layer Transformer reach in the semi-supervised setting?":
        "'Attention Is All You Need' paper ki Table 4 me semi-supervised setting me 4-layer Transformer ka WSJ 23 F1 kitna hai?",
    "Using Table 2 of the 'Attention Is All You Need' paper, how many BLEU points better is Transformer (big) than Transformer (base model) on English-to-French (41.8 minus 38.1, one decimal place)?":
        "'Attention Is All You Need' paper ki Table 2 se English-to-French me Transformer (big), Transformer (base model) se kitne BLEU points behtar hai (41.8 minus 38.1, ek decimal place)?",
    "In the 'Attention Is All You Need' paper, did replacing sinusoidal positional encodings with learned positional embeddings produce nearly identical results?":
        "'Attention Is All You Need' paper me sinusoidal positional encodings ko learned positional embeddings se badalne par kya lagbhag same results aaye?",
    "How many gigabytes of GPU memory did the Transformer (big) model need during training, according to the 'Attention Is All You Need' paper?":
        "'Attention Is All You Need' paper ke hisaab se Transformer (big) model ko training me kitne gigabytes GPU memory chahiye thi?",
    'How many total parameters does BERT-LARGE have in the BERT paper?':
        'BERT paper me BERT-LARGE ke total kitne parameters hain?',
    'What is the size of the WordPiece vocabulary used by BERT?':
        'BERT jo WordPiece vocabulary use karta hai uska size kitna hai?',
    'What accuracy does the final BERT pre-trained model achieve on the next sentence prediction (NSP) task?':
        'Final BERT pre-trained model next sentence prediction (NSP) task par kitni accuracy achieve karta hai?',
    'How long did each BERT pre-training run (BERT-BASE and BERT-LARGE) take to complete?':
        'BERT ka har pre-training run (BERT-BASE aur BERT-LARGE) complete hone me kitna time laga?',
    'What batch size was used to fine-tune BERT on SQuAD v2.0?':
        'BERT ko SQuAD v2.0 par fine-tune karne ke liye kitna batch size use hua?',
    'What batch size (in sequences) and how many training steps were used for BERT pre-training?':
        'BERT pre-training me kitna batch size (sequences me) aur kitne training steps use hue?',
    'What peak learning rate does BERT use for pre-training with Adam, and over how many initial steps is the learning rate warmed up?':
        'BERT pre-training me Adam ke saath peak learning rate kya hai, aur learning rate ko shuru ke kitne steps tak warm up kiya jata hai?',
    'In Table 1 of the BERT paper, what are the MNLI-(m/mm) test accuracies of BERT-LARGE?':
        'BERT paper ki Table 1 me BERT-LARGE ki MNLI-(m/mm) test accuracies kya hain?',
    'In Table 7 of the BERT paper (CoNLL-2003 NER), what Dev F1 does the feature-based approach that concatenates the last four hidden layers of BERT-BASE achieve?':
        'BERT paper ki Table 7 (CoNLL-2003 NER) me jo feature-based approach BERT-BASE ki last four hidden layers ko concatenate karta hai uska Dev F1 kitna hai?',
    'Using Table 1 of the BERT paper, how many points higher is the GLUE test Average of BERT-LARGE (82.1) than BERT-BASE (79.6), to one decimal place?':
        'BERT paper ki Table 1 se BERT-LARGE (82.1) ka GLUE test Average BERT-BASE (79.6) se kitne points zyada hai, ek decimal place tak?',
    "In the BERT paper's feature-based NER experiment on CoNLL-2003, is a CRF layer used in the output?":
        'BERT paper ke CoNLL-2003 feature-based NER experiment me kya output me CRF layer use hui hai?',
    'What accuracy does BERT-LARGE achieve on the XNLI cross-lingual benchmark?':
        'BERT-LARGE XNLI cross-lingual benchmark par kitni accuracy achieve karta hai?',
    'Which library does the RAG paper use to build its single MIPS index over the document embeddings?':
        'RAG paper document embeddings par apna single MIPS index banane ke liye kaun si library use karta hai?',
    'During RAG training, which component is kept fixed (not fine-tuned) so that the document index does not need to be periodically updated?':
        'RAG training me kaun sa component fixed rakha jata hai (fine-tune nahi hota) taaki document index ko baar baar update na karna pade?',
    "How many world leaders who had changed between 2016 and 2018 were used in the RAG paper's index hot-swapping experiment?":
        'RAG paper ke index hot-swapping experiment me 2016 aur 2018 ke beech badle hue kitne world leaders use hue?',
    'On Natural Questions, what accuracy does RAG achieve when the correct answer is not in any retrieved document?':
        'Natural Questions par jab sahi answer kisi bhi retrieved document me nahi hota, tab RAG kitni accuracy achieve karta hai?',
    "In the RAG paper's human evaluation for Jeopardy question generation, how many pairs of generations from BART and RAG-Token were assessed?":
        'RAG paper ki Jeopardy question generation ki human evaluation me BART aur RAG-Token ke kitne generation pairs assess hue?',
    'In the RAG paper, how many train, dev and test examples are in the SearchQA splits used for Jeopardy question generation?':
        'RAG paper me Jeopardy question generation ke liye use hue SearchQA splits me kitne train, dev aur test examples hain?',
    'For open-domain QA test results in the RAG paper, how many documents are retrieved for RAG-Token models and for RAG-Sequence models?':
        'RAG paper ke open-domain QA test results me RAG-Token models aur RAG-Sequence models ke liye kitne documents retrieve kiye jate hain?',
    'In Table 1 of the RAG paper, what exact match score does RAG-Sequence get on CuratedTrec (CT)?':
        'RAG paper ki Table 1 me RAG-Sequence ko CuratedTrec (CT) par kitna exact match score milta hai?',
    'In Table 2 of the RAG paper, what FEVER-3 label accuracy does RAG-Token reach?':
        'RAG paper ki Table 2 me RAG-Token ki FEVER-3 label accuracy kitni hai?',
    'Using Table 6 of the RAG paper, how many exact-match points does RAG-Sequence (44.0) improve over RAG-Sequence-Frozen (41.2) on Natural Questions, to one decimal place?':
        'RAG paper ki Table 6 se Natural Questions par RAG-Sequence (44.0), RAG-Sequence-Frozen (41.2) se kitne exact-match points behtar hai, ek decimal place tak?',
    'Does the RAG paper use supervision on retrieved evidence for its FEVER experiments?':
        'Kya RAG paper apne FEVER experiments me retrieved evidence par supervision use karta hai?',
    'What average retrieval latency in milliseconds per query does the RAG model achieve?':
        'RAG model ki average retrieval latency milliseconds per query me kitni hai?',
    "According to the GPT-4 technical report, when does the vast majority of GPT-4's pre-training data cut off?":
        'GPT-4 technical report ke hisaab se GPT-4 ka zyadatar pre-training data kab tak ka hai (cut off)?',
    "By how many percentage points does GPT-4 score higher than the latest GPT-3.5 on the GPT-4 report's internal adversarially-designed factuality evaluations?":
        'GPT-4 report ke internal adversarially-designed factuality evaluations me GPT-4, latest GPT-3.5 se kitne percentage points zyada score karta hai?',
    "In the GPT-4 technical report, on what percentage of 5,214 prompts were GPT-4's responses preferred over GPT-3.5's?":
        'GPT-4 technical report me 5,214 prompts me se kitne percent par GPT-4 ke responses GPT-3.5 ke responses se zyada pasand kiye gaye?',
    'On translated variants of MMLU, in how many of the 26 languages considered does GPT-4 surpass the English-language state-of-the-art?':
        'MMLU ke translated variants me, consider ki gayi 26 languages me se kitni me GPT-4 English-language state-of-the-art ko beat karta hai?',
    'In the GPT-4 technical report, how many experts from domains such as AI alignment, cybersecurity and biorisk were engaged to adversarially test the model?':
        'GPT-4 technical report me AI alignment, cybersecurity aur biorisk jaise domains ke kitne experts model ko adversarially test karne ke liye lagaye gaye?',
    'On the RealToxicityPrompts dataset, what percentage of generations are toxic for GPT-4 and for GPT-3.5 according to the GPT-4 technical report?':
        'GPT-4 technical report ke hisaab se RealToxicityPrompts dataset par GPT-4 aur GPT-3.5 ke kitne percent generations toxic hain?',
    'What expected calibration error (ECE) does the GPT-4 technical report show for the pre-trained model and for the post-trained (PPO) model on a subset of MMLU?':
        'GPT-4 technical report MMLU ke subset par pre-trained model aur post-trained (PPO) model ke liye expected calibration error (ECE) kitna dikhata hai?',
    'In Table 1 of the GPT-4 technical report, what LSAT score and percentile does GPT-4 achieve?':
        'GPT-4 technical report ki Table 1 me GPT-4 ko LSAT me kitna score aur kaun sa percentile milta hai?',
    'In Table 2 of the GPT-4 technical report, what accuracy does GPT-4 achieve on HellaSwag (10-shot)?':
        'GPT-4 technical report ki Table 2 me GPT-4 HellaSwag (10-shot) par kitni accuracy achieve karta hai?',
    "Using Table 2 of the GPT-4 technical report, how many percentage points higher is GPT-4's HumanEval score (67.0%) than GPT-3.5's (48.1%), to one decimal place?":
        'GPT-4 technical report ki Table 2 se GPT-4 ka HumanEval score (67.0%), GPT-3.5 (48.1%) se kitne percentage points zyada hai, ek decimal place tak?',
    'Did the GPT-4 technical report exclude BIG-bench from its reported results because portions of it were inadvertently mixed into the training set?':
        'Kya GPT-4 technical report ne BIG-bench ko apne reported results se is liye hata diya kyunki uska kuch hissa galti se training set me mil gaya tha?',
    'What is the maximum context length in tokens supported by GPT-4 according to the technical report?':
        'Technical report ke hisaab se GPT-4 ki maximum context length tokens me kitni hai?',
    'What is the total vocabulary size of the tokenizer used by Llama 2?':
        'Llama 2 jo tokenizer use karta hai uski total vocabulary size kitni hai?',
    'How many warmup steps does the cosine learning rate schedule use in Llama 2 pretraining?':
        'Llama 2 pretraining me cosine learning rate schedule kitne warmup steps use karta hai?',
    'What is the name of the method proposed in the Llama 2 paper to help the model keep following an initial instruction over multiple dialogue turns?':
        'Llama 2 paper me model ko kai dialogue turns tak pehli instruction follow karte rehne me madad karne wale method ka naam kya hai?',
    'In the Llama 2 paper, rejection sampling is performed only with which model size, with the smaller models fine-tuned on its samples?':
        'Llama 2 paper me rejection sampling sirf kis model size ke saath hota hai, jiske samples par chhote models fine-tune hote hain?',
    'In the Llama 2 pretraining data analysis, about what percentage of documents get a toxicity likelihood score of 0.5 or higher from the HateBERT classifier?':
        'Llama 2 ke pretraining data analysis me lagbhag kitne percent documents ko HateBERT classifier se 0.5 ya usse zyada toxicity likelihood score milta hai?',
    'For supervised fine-tuning of Llama 2-Chat, what batch size and sequence length (in tokens) are used?':
        'Llama 2-Chat ke supervised fine-tuning me kitna batch size aur sequence length (tokens me) use hota hai?',
    'What KL penalty beta values does Llama 2 use in PPO for the 7B and 13B models and for the 34B and 70B models?':
        'Llama 2 PPO me 7B aur 13B models ke liye aur 34B aur 70B models ke liye KL penalty beta values kya hain?',
    'In Table 6 of the Llama 2 paper, how many comparisons are in the Meta (Safety & Helpfulness) human preference data?':
        'Llama 2 paper ki Table 6 me Meta (Safety & Helpfulness) human preference data me kitne comparisons hain?',
    'In Table 3 of the Llama 2 paper, what AGI Eval score does Llama 2 70B get?':
        'Llama 2 paper ki Table 3 me Llama 2 70B ko AGI Eval me kitna score milta hai?',
    "Using Table 3 of the Llama 2 paper, how many points higher is Llama 2 70B's MMLU score (68.9) than Llama 1 65B's (63.4), to one decimal place?":
        'Llama 2 paper ki Table 3 se Llama 2 70B ka MMLU score (68.9), Llama 1 65B (63.4) se kitne points zyada hai, ek decimal place tak?',
    'Did the Llama 2 authors release the 34B parameter variants to the general public?':
        'Kya Llama 2 ke authors ne 34B parameter variants ko general public ke liye release kiya?',
    'What accuracy does Llama 2 70B achieve on the GPQA benchmark?':
        'Llama 2 70B GPQA benchmark par kitni accuracy achieve karta hai?',
    "In the survey 'A Survey of Large Language Models', how many diverse high-quality subsets is the Pile dataset constructed from?":
        "'A Survey of Large Language Models' survey me Pile dataset kitne diverse high-quality subsets se bana hai?",
    "In 'A Survey of Large Language Models', what values of beta1 and beta2 are commonly used as hyper-parameters for the Adam/AdamW optimizer when training LLMs?":
        "'A Survey of Large Language Models' me LLMs train karte time Adam/AdamW optimizer ke hyper-parameters ke roop me beta1 aur beta2 ki kaun si values aam taur par use hoti hain?",
    "In 'A Survey of Large Language Models', approximately how many instances does the HH-RLHF alignment dataset consist of?":
        "'A Survey of Large Language Models' me HH-RLHF alignment dataset me lagbhag kitne instances hain?",
    "In 'A Survey of Large Language Models', what context window length does Mistral NeMo (12B) feature?":
        "'A Survey of Large Language Models' me Mistral NeMo (12B) ki context window length kitni hai?",
    "In 'A Survey of Large Language Models', on up to how many tokens is Qwen2.5 pre-trained?":
        "'A Survey of Large Language Models' me Qwen2.5 ko kitne tokens tak pre-train kiya gaya hai?",
    "In 'A Survey of Large Language Models', how much text in total does the ROOTS corpus contain and how many languages does it cover?":
        "'A Survey of Large Language Models' me ROOTS corpus me kul kitna text hai aur wo kitni languages cover karta hai?",
    "In the survey's description of the Chinchilla scaling law (Hoffmann et al.), what lower and upper bounds of data size (in tokens) did they vary in their experiments?":
        'Survey me Chinchilla scaling law (Hoffmann et al.) ke description me unhone apne experiments me data size (tokens me) ki lower aur upper bound kya rakhi thi?',
    "In the survey's table of tuning and inference efficiency for LLaMA models, how long does LoRA tuning of LLaMA (65B) take on A800 GPUs?":
        'Survey ki LLaMA models ki tuning aur inference efficiency table me A800 GPUs par LLaMA (65B) ki LoRA tuning me kitna time lagta hai?',
    "In the survey's table of tuning and inference efficiency for LLaMA models, how many tokens per second does LLaMA (7B) achieve for 16-bit inference on a single A800 GPU?":
        'Survey ki LLaMA models ki tuning aur inference efficiency table me ek single A800 GPU par 16-bit inference me LLaMA (7B) kitne tokens per second deta hai?',
    "Using the survey's tuning-efficiency table for LLaMA on A800 GPUs, how many more hours does full tuning of LLaMA (65B, 11.2h) take than full tuning of LLaMA (7B, 3.0h)? Give one decimal place.":
        'Survey ki tuning-efficiency table (A800 GPUs par LLaMA) se full tuning of LLaMA (65B, 11.2h), full tuning of LLaMA (7B, 3.0h) se kitne ghante zyada leti hai? Ek decimal place me batao.',
    "According to 'A Survey of Large Language Models', is the Adafactor optimizer a variant of Adam designed to conserve GPU memory during training?":
        "'A Survey of Large Language Models' ke hisaab se kya Adafactor optimizer Adam ka ek variant hai jo training me GPU memory bachane ke liye bana hai?",
    "In 'A Survey of Large Language Models', what was the total training cost in US dollars of GPT-4?":
        "'A Survey of Large Language Models' me GPT-4 ki total training cost US dollars me kitni thi?",
    'In the Infosys Integrated Annual Report 2024-25, what was the free cash flow for fiscal 2025 (in Rs crore)?':
        'Infosys Integrated Annual Report 2024-25 me fiscal 2025 ka free cash flow (Rs crore me) kitna tha?',
    'In the Infosys Integrated Annual Report 2024-25, how many clients with more than US$50 million in revenue does Infosys have?':
        'Infosys Integrated Annual Report 2024-25 me Infosys ke kitne clients ka revenue US$50 million se zyada hai?',
    'In the Infosys Integrated Annual Report 2024-25, what was the large deal total contract value (TCV) in US$ billion?':
        'Infosys Integrated Annual Report 2024-25 me large deal total contract value (TCV) US$ billion me kitna tha?',
    'In the Infosys Integrated Annual Report 2024-25, how many people have received free learning programs through Infosys Springboard?':
        'Infosys Integrated Annual Report 2024-25 me Infosys Springboard ke zariye kitne logon ne free learning programs liye hain?',
    'In the Infosys Integrated Annual Report 2024-25, who is the Chairman of the Infosys board?':
        'Infosys Integrated Annual Report 2024-25 me Infosys board ke Chairman kaun hain?',
    "In the Infosys Integrated Annual Report 2024-25 financial capital highlights, what were Infosys's return on equity and its CRISIL rating?":
        'Infosys Integrated Annual Report 2024-25 ke financial capital highlights me Infosys ka return on equity aur CRISIL rating kya thi?',
    "In the Infosys Integrated Annual Report 2024-25, what was Infosys's consolidated cash and investments in Rs crore, and what was FCF conversion as a percentage of net profit?":
        'Infosys Integrated Annual Report 2024-25 me Infosys ka consolidated cash aur investments Rs crore me kitna tha, aur FCF conversion net profit ke percentage ke roop me kitna tha?',
    "In the key financial ratios table of the Infosys Integrated Annual Report 2024-25 (Management's discussion and analysis), what was the consolidated Price / Earnings (times) for 2025?":
        "Infosys Integrated Annual Report 2024-25 (Management's discussion and analysis) ki key financial ratios table me 2025 ke liye consolidated Price / Earnings (times) kitna tha?",
    'In the key financial ratios table of the Infosys Integrated Annual Report 2024-25, what was the consolidated interest coverage ratio (times) for 2025?':
        'Infosys Integrated Annual Report 2024-25 ki key financial ratios table me 2025 ke liye consolidated interest coverage ratio (times) kitna tha?',
    'In the key financial ratios table of the Infosys Integrated Annual Report 2024-25, by how much did consolidated Basic EPS in rupees for 2025 (64.50) exceed 2024 (63.39)? Give two decimal places.':
        'Infosys Integrated Annual Report 2024-25 ki key financial ratios table me 2025 ka consolidated Basic EPS rupees me (64.50), 2024 (63.39) se kitna zyada tha? Do decimal places me batao.',
    'Does the Infosys Integrated Annual Report 2024-25 describe Infosys Springboard as a free skilling platform?':
        'Kya Infosys Integrated Annual Report 2024-25 Infosys Springboard ko ek free skilling platform batati hai?',
    'How many GPUs did Infosys deploy in its own data centers in fiscal 2025 according to the Infosys Integrated Annual Report 2024-25?':
        'Infosys Integrated Annual Report 2024-25 ke hisaab se fiscal 2025 me Infosys ne apne data centers me kitne GPUs deploy kiye?',
    "In NVIDIA's fiscal 2026 annual report, how large was the charge associated with H20 excess inventory and purchase obligations?":
        'NVIDIA ki fiscal 2026 annual report me H20 excess inventory aur purchase obligations se jude charge ka size kitna tha?',
    "In NVIDIA's fiscal 2026 annual report, by what percentage did Gaming revenue increase in fiscal year 2026?":
        'NVIDIA ki fiscal 2026 annual report me fiscal year 2026 me Gaming revenue kitne percent badha?',
    "In NVIDIA's fiscal 2026 annual report, what was the Graphics segment revenue in fiscal 2026?":
        'NVIDIA ki fiscal 2026 annual report me fiscal 2026 me Graphics segment ka revenue kitna tha?',
    "In NVIDIA's 2026 proxy statement included in the annual report, on what date is the 2026 Annual Meeting of Stockholders?":
        'NVIDIA ke 2026 proxy statement me (jo annual report me shamil hai) 2026 Annual Meeting of Stockholders kis date ko hai?',
    "In NVIDIA's fiscal 2026 annual report, until what date do currently issued NVIDIA patents have expiration dates, i.e. the latest expiration (month and year)?":
        'NVIDIA ki fiscal 2026 annual report me abhi issued NVIDIA patents ki expiration dates kab tak hain, yaani sabse late expiration (mahina aur saal)?',
    "In NVIDIA's fiscal 2026 annual report, what percentages of total revenue did the two largest direct customers represent in fiscal 2026?":
        'NVIDIA ki fiscal 2026 annual report me fiscal 2026 me do sabse bade direct customers total revenue ka kitna kitna percentage the?',
    "In NVIDIA's fiscal 2026 annual report, in which years was NVIDIA incorporated in California and reincorporated in Delaware?":
        'NVIDIA ki fiscal 2026 annual report me NVIDIA kis saal California me incorporate hua aur kis saal Delaware me reincorporate hua?',
    "In NVIDIA's Consolidated Statements of Income for fiscal 2026, what was Research and development expense (in millions)?":
        'NVIDIA ke fiscal 2026 ke Consolidated Statements of Income me Research and development expense (millions me) kitna tha?',
    "In NVIDIA's Consolidated Statements of Income for fiscal 2026, what was income tax expense (in millions)?":
        'NVIDIA ke fiscal 2026 ke Consolidated Statements of Income me income tax expense (millions me) kitna tha?',
    "Using NVIDIA's fiscal 2026 Consolidated Statements of Income, how much larger was research and development expense (18,497) than sales, general and administrative expense (4,579), in millions?":
        'NVIDIA ke fiscal 2026 ke Consolidated Statements of Income se research and development expense (18,497), sales, general and administrative expense (4,579) se millions me kitna bada tha?',
    'Did NVIDIA report net income per diluted share above $4.50 for fiscal 2026 in its fiscal 2026 annual report?':
        'Kya NVIDIA ne apni fiscal 2026 annual report me fiscal 2026 ke liye net income per diluted share $4.50 se upar report kiya?',
    "In NVIDIA's fiscal 2026 annual report, how many Blackwell GPUs were shipped in total during fiscal 2026?":
        'NVIDIA ki fiscal 2026 annual report me fiscal 2026 me kul kitne Blackwell GPUs ship hue?',
    'In the NIST AI RMF Playbook, what does GOVERN 4.3 say organizational practices should enable?':
        'NIST AI RMF Playbook me GOVERN 4.3 ke hisaab se organizational practices ko kya enable karna chahiye?',
    'In the NIST AI RMF Playbook, what does GOVERN 1.7 require processes and procedures to be in place for?':
        'NIST AI RMF Playbook me GOVERN 1.7 kis cheez ke liye processes aur procedures hone ki requirement rakhta hai?',
    'In the NIST AI RMF Playbook, what does MANAGE 3.2 say should be monitored as part of regular monitoring and maintenance?':
        'NIST AI RMF Playbook me MANAGE 3.2 ke hisaab se regular monitoring aur maintenance ke hisse ke roop me kya monitor hona chahiye?',
    'In the NIST AI RMF Playbook, what does MAP 1.5 say must be determined and documented?':
        'NIST AI RMF Playbook me MAP 1.5 ke hisaab se kya determine aur document karna zaroori hai?',
    'In the NIST AI RMF Playbook, which kind of risk of the AI system does MEASURE 2.10 say is examined and documented?':
        'NIST AI RMF Playbook me MEASURE 2.10 ke hisaab se AI system ka kaun sa risk examine aur document kiya jata hai?',
    'In the NIST AI RMF Playbook, how does MANAGE 1.4 define negative residual risks?':
        'NIST AI RMF Playbook me MANAGE 1.4 negative residual risks ko kaise define karta hai?',
    'In the NIST AI RMF Playbook, which two properties of an AI system does MEASURE 2.7 say are evaluated and documented?':
        'NIST AI RMF Playbook me MEASURE 2.7 ke hisaab se AI system ki kaun si do properties evaluate aur document ki jati hain?',
    'In the NIST AI RMF Playbook, what does GOVERN 3.2 say policies and procedures should define and differentiate roles and responsibilities for?':
        'NIST AI RMF Playbook me GOVERN 3.2 ke hisaab se policies aur procedures ko kis cheez ke liye roles aur responsibilities define aur differentiate karne chahiye?',
    'In the NIST AI RMF Playbook, which two aspects of each identified impact does MAP 5.1 say are determined?':
        'NIST AI RMF Playbook me MAP 5.1 ke hisaab se har identified impact ke kaun se do aspects determine kiye jate hain?',
    'In the NIST AI RMF Playbook, does MEASURE 2.13 concern evaluating and documenting the effectiveness of the employed TEVV metrics and processes?':
        'Kya NIST AI RMF Playbook me MEASURE 2.13 employed TEVV metrics aur processes ki effectiveness ko evaluate aur document karne ke baare me hai?',
    'Which specific GPU model does the NIST AI RMF Playbook recommend for running TEVV benchmarks?':
        'NIST AI RMF Playbook TEVV benchmarks chalane ke liye kaun sa specific GPU model recommend karti hai?',
    'In the AWS Generative AI Lens, which sampling technique does GENOPS01-BP01 (Periodically evaluate functional performance) use together with custom metrics?':
        'AWS Generative AI Lens me GENOPS01-BP01 (Periodically evaluate functional performance) custom metrics ke saath kaun si sampling technique use karta hai?',
    'In the AWS Generative AI Lens, what threat does best practice GENSEC06-BP01 (data purification filters for model training workflows) address?':
        'AWS Generative AI Lens me best practice GENSEC06-BP01 (data purification filters for model training workflows) kis threat ko address karta hai?',
    'In the AWS Generative AI Lens, what is the level of risk exposed if best practice GENREL04-BP02 (Implement a model catalog) is not established?':
        'AWS Generative AI Lens me agar best practice GENREL04-BP02 (Implement a model catalog) establish na ho to exposed risk ka level kya hai?',
    'In the AWS Generative AI Lens, what is the level of risk exposed if best practice GENSEC04-BP01 (Implement a secure prompt catalog) is not established?':
        'AWS Generative AI Lens me agar best practice GENSEC04-BP01 (Implement a secure prompt catalog) establish na ho to exposed risk ka level kya hai?',
    "In the AWS Generative AI Lens, which best practice ID is titled 'Create stopping conditions to control long-running workflows'?":
        "AWS Generative AI Lens me kaun si best practice ID ka title 'Create stopping conditions to control long-running workflows' hai?",
    'In the AWS Generative AI Lens, which two best practices are listed under GENOPS04 (by topic)?':
        'AWS Generative AI Lens me GENOPS04 (by topic) ke neeche kaun si do best practices listed hain?',
    'In the AWS Generative AI Lens, what are the three best practices listed under GENREL05 (by topic)?':
        'AWS Generative AI Lens me GENREL05 (by topic) ke neeche kaun si teen best practices listed hain?',
    'In the AWS Generative AI Lens, what is the level of risk exposed if best practice GENPERF02-BP02 (Optimize inference parameters to improve response quality) is not established?':
        'AWS Generative AI Lens me agar best practice GENPERF02-BP02 (Optimize inference parameters to improve response quality) establish na ho to exposed risk ka level kya hai?',
    'In the AWS Generative AI Lens, what is the level of risk exposed if best practice GENSUS03-BP01 (Leverage smaller models and optimized inference techniques) is not established?':
        'AWS Generative AI Lens me agar best practice GENSUS03-BP01 (Leverage smaller models and optimized inference techniques) establish na ho to exposed risk ka level kya hai?',
    'In the AWS Generative AI Lens, is the level of risk exposed if GENREL03-BP02 (Implement timeout mechanisms on agentic workflows) is not established rated High?':
        'AWS Generative AI Lens me agar GENREL03-BP02 (Implement timeout mechanisms on agentic workflows) establish na ho to exposed risk ka level High rate kiya gaya hai kya?',
    'In the AWS Generative AI Lens, what is the per-1,000-token price of Amazon Bedrock Claude models?':
        'AWS Generative AI Lens me Amazon Bedrock Claude models ka per-1,000-token price kitna hai?',
    'In the AWS Well-Architected Machine Learning Lens, which two warm start types does Amazon SageMaker AI offer for hyperparameter tuning jobs?':
        'AWS Well-Architected Machine Learning Lens me Amazon SageMaker AI hyperparameter tuning jobs ke liye kaun se do warm start types offer karta hai?',
    'In the AWS Well-Architected Machine Learning Lens, up to how many parent jobs can be specified when configuring a SageMaker AI warm start tuning job?':
        'AWS Well-Architected Machine Learning Lens me SageMaker AI warm start tuning job configure karte time maximum kitne parent jobs specify kiye ja sakte hain?',
    'In the AWS Well-Architected Machine Learning Lens, which SageMaker AI capability is described as a compute reservation system for predictable access to high-demand GPU resources?':
        'AWS Well-Architected Machine Learning Lens me kaun si SageMaker AI capability ko high-demand GPU resources tak predictable access ke liye compute reservation system bataya gaya hai?',
    'According to the AWS Well-Architected Machine Learning Lens, which form of distributed training is designed for situations where the model is too large to fit on a single GPU?':
        'AWS Well-Architected Machine Learning Lens ke hisaab se distributed training ka kaun sa roop us situation ke liye bana hai jab model ek single GPU me fit hone ke liye bahut bada ho?',
    'In the AWS Well-Architected Machine Learning Lens, which training issues does Amazon SageMaker AI Debugger help identify, such as exploding tensors and what gradient problem?':
        'AWS Well-Architected Machine Learning Lens me Amazon SageMaker AI Debugger kaun se training issues pehchanne me madad karta hai, jaise exploding tensors aur kaun si gradient problem?',
    'In the AWS Well-Architected Machine Learning Lens, is SageMaker AI Serverless Inference suggested for workloads with unpredictable or intermittent traffic patterns?':
        'AWS Well-Architected Machine Learning Lens me kya SageMaker AI Serverless Inference unpredictable ya intermittent traffic patterns wale workloads ke liye suggest kiya gaya hai?',
    'In the AWS Well-Architected Machine Learning Lens, what is the on-demand hourly price of an ml.trn1.32xlarge instance?':
        'AWS Well-Architected Machine Learning Lens me ml.trn1.32xlarge instance ka on-demand hourly price kitna hai?',
    'Which AWS service does the AWS Well-Architected Machine Learning Lens recommend for automatically discovering and classifying sensitive data such as PII in S3 training data, using custom data identifiers if needed?':
        'AWS Well-Architected Machine Learning Lens S3 training data me PII jaisa sensitive data automatically discover aur classify karne ke liye (zarurat ho to custom data identifiers ke saath) kaun si AWS service recommend karti hai?',
    'In the AWS Well-Architected Machine Learning Lens (MLSUS05-BP02, Use efficient silicon), by how much can AWS Graviton3 reduce energy use compared to comparable Amazon EC2 instances?':
        'AWS Well-Architected Machine Learning Lens (MLSUS05-BP02, Use efficient silicon) me AWS Graviton3 comparable Amazon EC2 instances ke mukable energy use kitna kam kar sakta hai?',
    'In the AWS Well-Architected Machine Learning Lens, how much lower is the total energy consumption for training BERT Large from scratch on a Trainium-based cluster compared to comparable accelerated EC2 clusters?':
        'AWS Well-Architected Machine Learning Lens me BERT Large ko scratch se train karne par Trainium-based cluster me total energy consumption comparable accelerated EC2 clusters se kitna kam hai?',
    'In the AWS Well-Architected Machine Learning Lens (MLOPS01-BP02), what two kinds of SHAP baselines can you choose in SageMaker AI Clarify, differing in information content?':
        'AWS Well-Architected Machine Learning Lens (MLOPS01-BP02) me SageMaker AI Clarify me SHAP ke kaun se do baselines chun sakte hain, jo information content me alag hote hain?',
    'In the AWS Well-Architected Machine Learning Lens, what does best practice MLCOST04-BP09 recommend starting experimentation with, in order to iterate quickly at low cost?':
        'AWS Well-Architected Machine Learning Lens me best practice MLCOST04-BP09 kam cost me jaldi iterate karne ke liye experimentation kis cheez se shuru karne ko kehti hai?',
    'In Shakespeare\'s text, which character speaks the line "A horse! a horse! my kingdom for a horse!"?':
        "Shakespeare ke text me 'A horse! a horse! my kingdom for a horse!' line kaun sa character bolta hai?",
    'In Shakespeare\'s text, which character speaks the line "What\'s in a name? that which we call a rose By any other name would smell as sweet"?':
        "Shakespeare ke text me 'What's in a name? that which we call a rose By any other name would smell as sweet' line kaun sa character bolta hai?",
    'In Shakespeare\'s text, which character speaks the line "He jests at scars that never felt a wound."?':
        "Shakespeare ke text me 'He jests at scars that never felt a wound.' line kaun sa character bolta hai?",
    'In Shakespeare\'s text, which character says "I would they were barbarians--as they are, Though in Rome litter\'d--not Romans--as they are not"?':
        "Shakespeare ke text me 'I would they were barbarians--as they are, Though in Rome litter'd--not Romans--as they are not' kaun sa character kehta hai?",
    'In Shakespeare\'s text, which character begins the fable with the words "There was a time when all the body\'s members Rebell\'d against the belly, thus accused it"?':
        "Shakespeare ke text me kaun sa character 'There was a time when all the body's members Rebell'd against the belly, thus accused it' in shabdon se fable shuru karta hai?",
    'In Shakespeare\'s text, which character says "Every man\'s conscience is a thousand swords, To fight against that bloody homicide."?':
        "Shakespeare ke text me 'Every man's conscience is a thousand swords, To fight against that bloody homicide.' kaun sa character kehta hai?",
    'In Shakespeare\'s text, which character speaks immediately before King Richard III says "Tut, tut, thou art all ice, thy kindness freezeth"?':
        "Shakespeare ke text me King Richard III jab 'Tut, tut, thou art all ice, thy kindness freezeth' kehta hai usse theek pehle kaun sa character bolta hai?",
    'In Shakespeare\'s text, who says "Because my name is George.", and which character asks him "Upon what cause?" just before?':
        "Shakespeare ke text me 'Because my name is George.' kaun kehta hai, aur usse theek pehle kaun sa character use 'Upon what cause?' poochta hai?",
    'In Shakespeare\'s text, who says "I would they were barbarians--as they are", and which character speaks just before him with "Come, sir, along with us."?':
        "Shakespeare ke text me 'I would they were barbarians--as they are' kaun kehta hai, aur usse theek pehle kaun sa character 'Come, sir, along with us.' bolta hai?",
    'In Shakespeare\'s text, does Juliet speak the line "O Romeo, Romeo! wherefore art thou Romeo?"':
        "Shakespeare ke text me kya Juliet 'O Romeo, Romeo! wherefore art thou Romeo?' line bolti hai?",
    'In Shakespeare\'s text, which character speaks the line "To be, or not to be, that is the question"?':
        "Shakespeare ke text me 'To be, or not to be, that is the question' line kaun sa character bolta hai?",
    'How many times does the speaker label "MENENIUS:" appear in the Shakespeare text?':
        'Shakespeare ke text me speaker label "MENENIUS:" kitni baar aata hai?',
    'How many spaces per indentation level does PEP 8 recommend?':
        'PEP 8 har indentation level ke liye kitne spaces recommend karta hai?',
    'In PEP 8, how many blank lines should surround top-level function and class definitions?':
        'PEP 8 me top-level function aur class definitions ke aas paas kitni blank lines honi chahiye?',
    "According to PEP 8, what is generally better to do when a function argument's name clashes with a reserved keyword, such as writing class_ rather than clss?":
        'PEP 8 ke hisaab se jab kisi function argument ka naam reserved keyword se takrata hai, tab aam taur par kya karna behtar hai, jaise clss likhne ki jagah class_ likhna?',
    "According to PEP 8's programming recommendations, which two exceptions does a bare except: clause catch that make it harder to interrupt a program with Control-C?":
        'PEP 8 ki programming recommendations ke hisaab se bare except: clause kaun si do exceptions pakadta hai jisse program ko Control-C se rokna mushkil ho jata hai?',
    'In PEP 8, which two string methods should be used instead of string slicing to check for prefixes or suffixes?':
        'PEP 8 me prefixes ya suffixes check karne ke liye string slicing ki jagah kaun se do string methods use karne chahiye?',
    'According to PEP 8, which function should always be used for object type comparisons instead of comparing types directly?':
        'PEP 8 ke hisaab se object type comparisons ke liye types ko seedha compare karne ki jagah hamesha kaun sa function use karna chahiye?',
    'In PEP 8, for a class FooBar, what does the class attribute __boo become after name mangling, and which kind of leading-underscore naming invokes it?':
        'PEP 8 me class FooBar ke liye class attribute __boo name mangling ke baad kya ban jata hai, aur kis tarah ka leading-underscore naming ise invoke karta hai?',
    'According to PEP 8, which source file encoding should code in the core Python distribution always use?':
        'PEP 8 ke hisaab se core Python distribution ke code ko hamesha kaun sa source file encoding use karna chahiye?',
    'According to PEP 8, by at least how many spaces should an inline comment be separated from the statement?':
        'PEP 8 ke hisaab se inline comment ko statement se kam se kam kitne spaces ki doori par rakhna chahiye?',
    "For new code, does PEP 8 suggest breaking lines before or after binary operators (Knuth's style)?":
        "Naye code ke liye kya PEP 8 binary operators se pehle ya baad me line todne ko kehta hai (Knuth's style)?",
    'Does PEP 8 allow mixing tabs and spaces for indentation in Python?':
        'Kya PEP 8 Python me indentation ke liye tabs aur spaces ko mix karne ki ijazat deta hai?',
    'What maximum number of parameters does PEP 8 allow for a single function definition?':
        'PEP 8 ek single function definition me maximum kitne parameters ki ijazat deta hai?',
    'In the Flask quickstart, which HTML form attribute must be set to multipart/form-data so that the browser transmits uploaded files?':
        'Flask quickstart me kaun si HTML form attribute multipart/form-data set karni padti hai taaki browser uploaded files bhej sake?',
    'In the Flask quickstart, which Werkzeug function should you pass a client-supplied filename through before using it to store a file on the server?':
        'Flask quickstart me server par file store karne se pehle client ke diye hue filename ko kaun se Werkzeug function se pass karna chahiye?',
    "In the Flask quickstart's table of URL converter types, which converter is described as like string but also accepting slashes?":
        "Flask quickstart ki URL converter types ki table me kaun sa converter 'string jaisa hai par slashes bhi accept karta hai' bataya gaya hai?",
    'In the Flask quickstart, what does the 401 status code in the abort(401) example mean?':
        'Flask quickstart me abort(401) example me 401 status code ka kya matlab hai?',
    'In the Flask quickstart, if an error handler returns a template without specifying a status code, what status code is assumed by default?':
        'Flask quickstart me agar error handler status code batae bina template return kare to default me kaun sa status code maana jata hai?',
    'In the Flask quickstart, what name should you avoid giving your application file because it would conflict with Flask itself?':
        'Flask quickstart me apni application file ka kaun sa naam nahi rakhna chahiye kyunki wo Flask se hi conflict karega?',
    'In the Flask quickstart, which function can be used to retrieve flashed messages, and which is also available in templates?':
        'Flask quickstart me flashed messages retrieve karne ke liye kaun sa function use ho sakta hai, aur kaun sa templates me bhi available hai?',
    'In the Flask quickstart, if your application file is named a certain way you do not need the --app option. Which two file names are mentioned?':
        'Flask quickstart me agar application file ka naam ek khaas tarah ka ho to --app option ki zarurat nahi padti. Kaun se do file names mention hain?',
    "In the Flask quickstart's About Responses section, what status code and mimetype are used when a view returns a string?":
        'Flask quickstart ke About Responses section me jab view string return karta hai to kaun sa status code aur mimetype use hota hai?',
    "In the Flask quickstart, if a route is defined as '/about' without a trailing slash, does accessing '/about/' produce a 404 Not Found error?":
        "Flask quickstart me agar route '/about' bina trailing slash ke define ho, to kya '/about/' kholne par 404 Not Found error aata hai?",
    'In the Flask quickstart, how do you enable asynchronous async def view functions?':
        'Flask quickstart me asynchronous async def view functions ko kaise enable karte hain?',
    "In the Flask quickstart, how is a view's return value treated if it is an iterator or generator returning strings or bytes?":
        'Flask quickstart me agar view ka return value strings ya bytes dene wala iterator ya generator ho, to use kaise treat kiya jata hai?',
    'In PEP 484, what two arguments does NewType accept?':
        'PEP 484 me NewType kaun se do arguments leta hai?',
    'In PEP 484, what value does typing.TYPE_CHECKING have at runtime?':
        'PEP 484 me typing.TYPE_CHECKING ki runtime par kya value hoti hai?',
    'In PEP 484, what does a cast(t, x) call do at runtime to the expression x?':
        'PEP 484 me cast(t, x) call runtime par expression x ke saath kya karta hai?',
    'In PEP 484, by convention, which suffix should names of type variables defined with covariant=True end in?':
        'PEP 484 me convention ke hisaab se covariant=True se define kiye gaye type variables ke naam kis suffix par khatam hone chahiye?',
    'In PEP 484, which special type annotates functions that never return normally, and which function from the sys module is given as an example of such a function?':
        'PEP 484 me jo functions normally kabhi return nahi karte unhe annotate karne ke liye kaun sa special type hai, aur sys module ka kaun sa function aise function ke example ke roop me diya gaya hai?',
    'In PEP 484, what is plain Type without brackets equivalent to?':
        'PEP 484 me bina brackets ka plain Type kis ke barabar hai?',
    'In PEP 484, what is the effect of a # type: ignore comment placed on a line by itself at the top of a file, before any docstrings or imports?':
        'PEP 484 me file ke sabse upar, docstrings ya imports se pehle, akeli line par # type: ignore comment lagane ka kya effect hota hai?',
    'In PEP 484, what must the argument to TypeVar() be, relative to the variable the TypeVar is assigned to?':
        'PEP 484 me TypeVar() ka argument us variable ke relation me kya hona chahiye jise TypeVar assign kiya gaya hai?',
    'In PEP 484, is the class definition class Pair(Generic[T, T]) valid?':
        'PEP 484 me kya class definition class Pair(Generic[T, T]) valid hai?',
    'In PEP 484, which two types is the type variable AnyStr constrained to in the concat example?':
        'PEP 484 ke concat example me type variable AnyStr kaun se do types tak constrained hai?',
    'In PEP 484, the NoReturn type is only valid as what kind of annotation?':
        'PEP 484 me NoReturn type sirf kis tarah ke annotation ke roop me valid hai?',
    'What does PEP 484 say about defining structural subtypes with TypedDict classes?':
        'PEP 484 TypedDict classes se structural subtypes define karne ke baare me kya kehta hai?',
    "In the Titanic dataset, what is the fare paid by passenger 'Allen, Mr. William Henry'?":
        "Titanic dataset me passenger 'Allen, Mr. William Henry' ne kitna fare diya?",
    "In the Titanic dataset, what is the ticket number of passenger 'Astor, Mrs. John Jacob (Madeleine Talmadge Force)'?":
        "Titanic dataset me passenger 'Astor, Mrs. John Jacob (Madeleine Talmadge Force)' ka ticket number kya hai?",
    "In the Titanic dataset, what is the Cabin of passenger 'Porter, Mr. Walter Chamberlain'?":
        "Titanic dataset me passenger 'Porter, Mr. Walter Chamberlain' ka Cabin kya hai?",
    "In the Titanic dataset, what is the Age of passenger 'Bateman, Rev. Robert James'?":
        "Titanic dataset me passenger 'Bateman, Rev. Robert James' ki Age kitni hai?",
    'In the Titanic dataset, how many passengers embarked at Southampton (Embarked = S)?':
        'Titanic dataset me kitne passengers Southampton (Embarked = S) se chadhe?',
    'In the Titanic dataset, how many passengers have a recorded Cabin value (non-empty)?':
        'Titanic dataset me kitne passengers ki Cabin value recorded hai (khali nahi hai)?',
    'In the Titanic dataset, how many passengers have a missing Age value?':
        'Titanic dataset me kitne passengers ki Age value missing hai?',
    'In the Titanic dataset, how many passengers were travelling with at least one parent or child (Parch greater than 0)?':
        'Titanic dataset me kitne passengers kam se kam ek parent ya child ke saath travel kar rahe the (Parch 0 se zyada)?',
    'In the Titanic dataset, what is the average Fare across all passengers, rounded to 2 decimal places?':
        'Titanic dataset me sabhi passengers ka average Fare kitna hai, 2 decimal places tak round karke?',
    'In the Titanic dataset, what percentage of passengers are male, rounded to 1 decimal place?':
        'Titanic dataset me kitne percent passengers male hain, 1 decimal place tak round karke?',
    'In the Titanic dataset, what is the average Age of passengers who survived (Survived = 1), ignoring missing values, rounded to 1 decimal place?':
        'Titanic dataset me jo passengers survive hue (Survived = 1) unki average Age kitni hai, missing values ko ignore karke, 1 decimal place tak round karke?',
    'In the Titanic dataset, what is the average Fare paid by first-class passengers (Pclass = 1), rounded to the nearest whole number?':
        'Titanic dataset me first-class passengers (Pclass = 1) ka average Fare kitna hai, nearest whole number tak round karke?',
    'In the Titanic dataset, is the oldest passenger (by Age) female?':
        'Titanic dataset me kya sabse bujurg passenger (Age ke hisaab se) female hai?',
    'In the Titanic dataset, what lifeboat number was each survivor rescued in?':
        'Titanic dataset me har survivor ko kaun si lifeboat number me bachaya gaya tha?',
    'In the tips dataset, what is the total_bill in data row 50 (counting the first record after the header as row 1)?':
        'Tips dataset me data row 50 ka total_bill kitna hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the tips dataset, what is the tip in data row 6 (counting the first record after the header as row 1)?':
        'Tips dataset me data row 6 ki tip kitni hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the tips dataset, what is the total_bill of the record with the largest tip?':
        'Tips dataset me sabse badi tip wale record ka total_bill kitna hai?',
    'In the tips dataset, what is the day of the week in data row 150 (counting the first record after the header as row 1)?':
        'Tips dataset me data row 150 me hafte ka din kaun sa hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the tips dataset, how many records are for smokers (smoker = Yes)?':
        'Tips dataset me kitne records smokers (smoker = Yes) ke hain?',
    'In the tips dataset, how many records are for Lunch (time = Lunch)?':
        'Tips dataset me kitne records Lunch (time = Lunch) ke hain?',
    'In the tips dataset, what is the total of all tips, rounded to 2 decimal places?':
        'Tips dataset me saari tips ka total kitna hai, 2 decimal places tak round karke?',
    'In the tips dataset, how many records are for a party size of exactly 6?':
        'Tips dataset me kitne records exactly 6 party size ke hain?',
    'In the tips dataset, what is the average total_bill, rounded to 2 decimal places?':
        'Tips dataset me average total_bill kitna hai, 2 decimal places tak round karke?',
    'In the tips dataset, what is the average party size (size), rounded to 2 decimal places?':
        'Tips dataset me average party size (size) kitni hai, 2 decimal places tak round karke?',
    'In the tips dataset, what percentage of records are for female customers (sex = Female), rounded to 1 decimal place?':
        'Tips dataset me kitne percent records female customers (sex = Female) ke hain, 1 decimal place tak round karke?',
    'In the tips dataset, what is the average tip for Saturday records (day = Sat), rounded to 2 decimal places?':
        'Tips dataset me Saturday records (day = Sat) ki average tip kitni hai, 2 decimal places tak round karke?',
    'In the tips dataset, is the average tip for Dinner records higher than the average tip for Lunch records?':
        'Tips dataset me kya Dinner records ki average tip Lunch records ki average tip se zyada hai?',
    'In the tips dataset, what payment method (cash or card) was used for the table with the largest party size?':
        'Tips dataset me sabse badi party size wali table ne kaun sa payment method (cash ya card) use kiya?',
    'In the penguins dataset, what is the species in data row 200 (counting the first record after the header as row 1)?':
        'Penguins dataset me data row 200 ki species kya hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the penguins dataset, what is the body_mass_g in data row 300 (counting the first record after the header as row 1)?':
        'Penguins dataset me data row 300 ka body_mass_g kitna hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the penguins dataset, what is the island in data row 100 (counting the first record after the header as row 1)?':
        'Penguins dataset me data row 100 ka island kaun sa hai (header ke baad ke pehle record ko row 1 gino)?',
    'In the penguins dataset, which species has the penguin with the lowest body_mass_g, and what is that mass (ignoring missing values)?':
        'Penguins dataset me sabse kam body_mass_g wala penguin kis species ka hai, aur wo mass kitna hai (missing values ignore karke)?',
    'In the penguins dataset, how many Chinstrap penguins are there?':
        'Penguins dataset me kitne Chinstrap penguins hain?',
    'In the penguins dataset, how many penguins have a missing sex value?':
        'Penguins dataset me kitne penguins ki sex value missing hai?',
    'In the penguins dataset, how many penguins were recorded on Dream island?':
        'Penguins dataset me kitne penguins Dream island par record hue?',
    'In the penguins dataset, what is the longest flipper_length_mm recorded?':
        'Penguins dataset me sabse lambi flipper_length_mm kitni record hui hai?',
    'In the penguins dataset, what is the average flipper_length_mm of Adelie penguins, ignoring missing values, rounded to 1 decimal place?':
        'Penguins dataset me Adelie penguins ki average flipper_length_mm kitni hai, missing values ignore karke, 1 decimal place tak round karke?',
    'In the penguins dataset, what is the average bill_length_mm across all penguins, ignoring missing values, rounded to 1 decimal place?':
        'Penguins dataset me sabhi penguins ki average bill_length_mm kitni hai, missing values ignore karke, 1 decimal place tak round karke?',
    'In the penguins dataset, by how many grams does the average body_mass_g of Gentoo penguins exceed that of Adelie penguins (ignoring missing values), rounded to the nearest whole number?':
        'Penguins dataset me Gentoo penguins ka average body_mass_g Adelie penguins se kitne grams zyada hai (missing values ignore karke), nearest whole number tak round karke?',
    'In the penguins dataset, what is the average bill_depth_mm of Chinstrap penguins, rounded to 2 decimal places?':
        'Penguins dataset me Chinstrap penguins ki average bill_depth_mm kitni hai, 2 decimal places tak round karke?',
    'In the penguins dataset, are there more male than female penguins (ignoring missing values for sex)?':
        'Penguins dataset me kya male penguins female se zyada hain (sex ki missing values ignore karke)?',
    'In the penguins dataset, in which year was each penguin recorded?':
        'Penguins dataset me har penguin kis saal record hua tha?',
    "In the car mpg dataset, what is the weight of the car named 'dodge challenger se'?":
        "Car mpg dataset me 'dodge challenger se' naam ki car ka weight kitna hai?",
    "In the car mpg dataset, what is the mpg of the car named 'datsun 510 (sw)'?":
        "Car mpg dataset me 'datsun 510 (sw)' naam ki car ka mpg kitna hai?",
    "In the car mpg dataset, what is the acceleration of the car named 'ford mustang cobra'?":
        "Car mpg dataset me 'ford mustang cobra' naam ki car ka acceleration kitna hai?",
    "In the car mpg dataset, what is the weight of the car named 'datsun 510'?":
        "Car mpg dataset me 'datsun 510' naam ki car ka weight kitna hai?",
    'In the car mpg dataset, how many cars have 8 cylinders?':
        'Car mpg dataset me kitni cars me 8 cylinders hain?',
    'In the car mpg dataset, how many cars have origin europe?':
        'Car mpg dataset me kitni cars ka origin europe hai?',
    'In the car mpg dataset, how many cars have a missing horsepower value?':
        'Car mpg dataset me kitni cars ki horsepower value missing hai?',
    'In the car mpg dataset, how many cars have model_year 76?':
        'Car mpg dataset me kitni cars ka model_year 76 hai?',
    'In the car mpg dataset, what is the average weight of cars with 4 cylinders, rounded to the nearest whole number?':
        'Car mpg dataset me 4 cylinders wali cars ka average weight kitna hai, nearest whole number tak round karke?',
    'In the car mpg dataset, what is the average horsepower of all cars, ignoring missing values, rounded to 1 decimal place?':
        'Car mpg dataset me sabhi cars ki average horsepower kitni hai, missing values ignore karke, 1 decimal place tak round karke?',
    'In the car mpg dataset, what is the average mpg of cars with origin usa, rounded to 2 decimal places?':
        'Car mpg dataset me origin usa wali cars ka average mpg kitna hai, 2 decimal places tak round karke?',
    'In the car mpg dataset, what percentage of cars have origin usa, rounded to 1 decimal place?':
        'Car mpg dataset me kitne percent cars ka origin usa hai, 1 decimal place tak round karke?',
    'In the car mpg dataset, is the average mpg of cars from model_year 80 higher than the average mpg of cars from model_year 70?':
        'Car mpg dataset me kya model_year 80 ki cars ka average mpg model_year 70 ki cars ke average mpg se zyada hai?',
    "In the car mpg dataset, what is the price of the car named 'dodge challenger se'?":
        "Car mpg dataset me 'dodge challenger se' naam ki car ki price kitni hai?",
    'In the airline_passengers dataset, how many passengers were recorded in March 1953?':
        'Airline_passengers dataset me March 1953 me kitne passengers record hue?',
    'In the airline_passengers dataset, how many passengers were recorded in August 1958?':
        'Airline_passengers dataset me August 1958 me kitne passengers record hue?',
    'In the airline_passengers dataset, how many passengers were recorded in December 1951?':
        'Airline_passengers dataset me December 1951 me kitne passengers record hue?',
    'In the airline_passengers dataset, how many passengers were recorded in May 1956?':
        'Airline_passengers dataset me May 1956 me kitne passengers record hue?',
    'In the airline_passengers dataset, what is the total number of passengers recorded in 1955?':
        'Airline_passengers dataset me 1955 me record hue passengers ki total sankhya kitni hai?',
    'In the airline_passengers dataset, what is the total number of passengers recorded in 1960?':
        'Airline_passengers dataset me 1960 me record hue passengers ki total sankhya kitni hai?',
    'In the airline_passengers dataset, in how many months were more than 400 passengers recorded?':
        'Airline_passengers dataset me kitne mahino me 400 se zyada passengers record hue?',
    'In the airline_passengers dataset, what is the highest number of passengers recorded in any month of 1952?':
        'Airline_passengers dataset me 1952 ke kisi bhi mahine me sabse zyada kitne passengers record hue?',
    'In the airline_passengers dataset, what is the average monthly number of passengers in 1954, rounded to 1 decimal place?':
        'Airline_passengers dataset me 1954 me average monthly passengers kitne the, 1 decimal place tak round karke?',
    'In the airline_passengers dataset, what is the average monthly number of passengers across the whole dataset, rounded to 1 decimal place?':
        'Airline_passengers dataset me poore dataset me average monthly passengers kitne hain, 1 decimal place tak round karke?',
    'In the airline_passengers dataset, by how many passengers did December 1960 exceed December 1949?':
        'Airline_passengers dataset me December 1960 December 1949 se kitne passengers zyada tha?',
    'In the airline_passengers dataset, what is the percentage increase in monthly passengers from January 1949 to January 1960, rounded to 1 decimal place?':
        'Airline_passengers dataset me January 1949 se January 1960 tak monthly passengers me kitne percent ki badhotri hui, 1 decimal place tak round karke?',
    'In the airline_passengers dataset, were more passengers recorded in August 1957 than in July 1957?':
        'Airline_passengers dataset me kya August 1957 me July 1957 se zyada passengers record hue?',
    'In the airline_passengers dataset, which airline carried the most passengers in 1958?':
        'Airline_passengers dataset me 1958 me kaun si airline ne sabse zyada passengers carry kiye?',
    'In the tips dataset, what is the largest total_bill recorded?':
        'Tips dataset me sabse badi total_bill kitni record hui hai?',
    'In the penguins dataset, how many Adelie penguins are there?':
        'Penguins dataset me kitne Adelie penguins hain?',
}