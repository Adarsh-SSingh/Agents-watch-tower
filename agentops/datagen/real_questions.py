"""Questions about REAL public documents (Phase 2: generalisation check).

Each row: (doc file, doc type, question type, question, expected pieces, location)
  expected pieces  "|" = every piece must appear in the answer (AND)
                   "~" = alternative spellings of ONE piece (OR)
  location         page number (PDF / reports) or section / line / "computed" (not judged)
Answers were checked against the real files. CSV answers are computed with pandas.
"""

QUESTIONS = [
    # ---- attention_paper.pdf
    ("attention_paper.pdf", "Research Paper", "Fact", "What BLEU score did the Transformer (big) achieve on the WMT 2014 English-to-German translation task?", "28.4", "p8"),
    ("attention_paper.pdf", "Research Paper", "Fact", "How many identical layers does the Transformer encoder stack contain?", "6", "p3"),
    ("attention_paper.pdf", "Research Paper", "Multi", "How many parallel attention heads does the Transformer use, and what are the dimensions dk and dv of each head?", "8|64", "p5"),
    ("attention_paper.pdf", "Research Paper", "Fact", "How many warmup steps are used in the Transformer learning-rate schedule?", "4000", "p7"),
    ("attention_paper.pdf", "Research Paper", "Unanswerable", "What GLUE score did the Transformer model achieve?", "", "-"),
    # ---- bert_paper.pdf
    ("bert_paper.pdf", "Research Paper", "Fact", "How many parameters does BERT-BASE have?", "110m~110 million", "p3"),
    ("bert_paper.pdf", "Research Paper", "Fact", "What percentage of WordPiece tokens does BERT mask at random in its masked language model task?", "15", "p4"),
    ("bert_paper.pdf", "Research Paper", "Fact", "What GLUE score does BERT report in its abstract?", "80.5", "p1"),
    ("bert_paper.pdf", "Research Paper", "Multi", "Which two corpora is BERT pre-trained on?", "bookscorpus|wikipedia", "p5"),
    ("bert_paper.pdf", "Research Paper", "Unanswerable", "What score did BERT achieve on the HumanEval coding benchmark?", "", "-"),
    # ---- rag_paper.pdf
    ("rag_paper.pdf", "Research Paper", "Multi", "Which retriever and which generator does RAG combine?", "dpr|bart", "p2"),
    ("rag_paper.pdf", "Research Paper", "Multi", "Into how many words is each Wikipedia article chunked, and how many documents does that give in total?", "100|21m~21 million", "p4"),
    ("rag_paper.pdf", "Research Paper", "Fact", "Which Wikipedia dump does the RAG paper use?", "december 2018~dec 2018", "p4"),
    ("rag_paper.pdf", "Research Paper", "Table Lookup", "What exact-match score does RAG-Sequence get on Natural Questions?", "44.5", "p6"),
    ("rag_paper.pdf", "Research Paper", "Fact", "How many parameters does the BART-large generator have?", "400m~400 million", "p3"),
    # ---- gpt4_technical_report.pdf
    ("gpt4_technical_report.pdf", "Technical Report", "Table Lookup", "What score did GPT-4 get on the simulated Uniform Bar Exam and in which percentile?", "298|90th~90 th~top 10", "p5"),
    ("gpt4_technical_report.pdf", "Technical Report", "Table Lookup", "What accuracy does GPT-4 report on MMLU?", "86.4", "p7"),
    ("gpt4_technical_report.pdf", "Technical Report", "Yes No", "Does the GPT-4 technical report disclose the model size and architecture details?", "no", "p2"),
    ("gpt4_technical_report.pdf", "Technical Report", "Fact", "By how much did GPT-4 reduce its tendency to respond to requests for disallowed content compared to GPT-3.5?", "82", "p13"),
    # ---- llama2_paper.pdf
    ("llama2_paper.pdf", "Research Paper", "Fact", "How many tokens was Llama 2 pretrained on?", "2 trillion~2t", "p5"),
    ("llama2_paper.pdf", "Research Paper", "Fact", "Which attention technique did Llama 2 adopt to improve inference scalability of larger models?", "grouped-query~grouped query~gqa", "p5"),
    ("llama2_paper.pdf", "Research Paper", "Fact", "How many supervised fine-tuning annotations did Llama 2-Chat use?", "27540", "p9"),
    ("llama2_paper.pdf", "Research Paper", "Multi", "How many GPU hours did Llama 2 pretraining use and what were the estimated emissions?", "3.3m~3.3 million|539", "p7"),
    # ---- llm_survey.pdf.txt
    ("llm_survey.pdf", "Survey", "Multi", "Which three typical emergent abilities of LLMs does the survey introduce?", "in-context learning~in context learning|instruction following|step-by-step reasoning~step by step reasoning", "sec 2.1"),
    ("llm_survey.pdf", "Survey", "Fact", "Which GPT model exhibited a strong in-context learning ability, according to the survey?", "175b|gpt-3", "sec 2.1"),
    ("llm_survey.pdf", "Survey", "Fact", "How does the survey define emergent abilities of LLMs?", "small models|large models", "sec 2.1"),
    ("llm_survey.pdf", "Survey", "Multi", "Which three model sizes is PaLM often tested with?", "8b|62b|540b", "sec 2.1"),
    # ---- infosys_annual_report.pdf.txt
    ("infosys_annual_report.pdf", "Annual Report", "Fact", "What were Infosys total revenues in fiscal 2025?", "162990", "p19"),
    ("infosys_annual_report.pdf", "Annual Report", "Fact", "How many employees does Infosys report (permanent employees)?", "323578", "p19"),
    ("infosys_annual_report.pdf", "Annual Report", "Fact", "What total dividend per share did Infosys announce, including the interim dividend?", "43", "p33"),
    ("infosys_annual_report.pdf", "Annual Report", "Multi", "When and where was Infosys incorporated, and under what name?", "pune|1981|infosys consultants", "p137"),
    ("infosys_annual_report.pdf", "Annual Report", "Unanswerable", "What was Infosys revenue in fiscal 2030?", "", "-"),
    # ---- nvidia_annual_report_2026.pdf.txt
    ("nvidia_annual_report_2026.pdf", "Annual Report", "Multi", "What was NVIDIA total revenue in fiscal 2026 and how much did it grow?", "215.9|65", "Business overview"),
    ("nvidia_annual_report_2026.pdf", "Annual Report", "Fact", "What was NVIDIA gross margin in fiscal 2026?", "71.1", "Business overview"),
    ("nvidia_annual_report_2026.pdf", "Annual Report", "Multi", "How many employees did NVIDIA have at the end of fiscal 2026 and in how many countries?", "42000|38", "Item 1, Human Capital"),
    ("nvidia_annual_report_2026.pdf", "Annual Report", "Fact", "By how much did Data Center networking revenue increase in fiscal 2026?", "142", "Business overview"),
    # ---- NIST_AI_RMF_Playbook.pdf.txt
    ("NIST_AI_RMF_Playbook.pdf", "Framework", "Fact", "What does GOVERN 1.1 of the NIST AI RMF Playbook say?", "legal|regulatory|documented", "p4"),
    ("NIST_AI_RMF_Playbook.pdf", "Framework", "Fact", "What does GOVERN 1.6 require?", "inventory", "p12"),
    ("NIST_AI_RMF_Playbook.pdf", "Framework", "Fact", "What does MANAGE 4.1 cover?", "post-deployment~post deployment|monitoring", "p53"),
    ("NIST_AI_RMF_Playbook.pdf", "Framework", "Fact", "What does MEASURE 2.6 require?", "safety", "p108"),
    # ---- AWS_Generative_AI_Lens.pdf.txt
    ("AWS_Generative_AI_Lens.pdf", "Framework", "Multi", "Which six pillars does the Generative AI Lens use?", "operational excellence|security|reliability|performance efficiency|cost optimization|sustainability", "p20"),
    ("AWS_Generative_AI_Lens.pdf", "Framework", "Fact", "Which best practice covers enabling tracing for agents and RAG workflows?", "genops03-bp02", "p106"),
    ("AWS_Generative_AI_Lens.pdf", "Framework", "Multi", "Which related tools are listed for tracing agents and RAG workflows?", "cloudwatch|opentelemetry|langfuse", "p110"),
    ("AWS_Generative_AI_Lens.pdf", "Framework", "Fact", "What is best practice GENSEC01-BP01?", "least privilege", "p124"),
    # ---- AWS_Machine_Learning_Lens.pdf.txt
    ("AWS_Machine_Learning_Lens.pdf", "Framework", "Multi", "Which phases does the Well-Architected machine learning lifecycle include?", "business goal|problem framing|data processing|model development|deployment|monitoring", "p5"),
    ("AWS_Machine_Learning_Lens.pdf", "Framework", "Yes No", "Are the phases of the ML lifecycle necessarily sequential?", "no", "p5"),
    ("AWS_Machine_Learning_Lens.pdf", "Framework", "Fact", "What does the model monitoring phase do according to the ML Lens?", "performance|early detection", "p7"),
    ("AWS_Machine_Learning_Lens.pdf", "Framework", "Fact", "What is best practice MLOPS02-BP04?", "lineage", "p46"),
    # ---- tiny_shakespeare.txt
    ("tiny_shakespeare.txt", "Literature", "Fact", "Which character speaks first in the text?", "first citizen", "line 1"),
    ("tiny_shakespeare.txt", "Literature", "Fact", "Who speaks the line \"Now is the winter of our discontent\"?", "gloucester", "line 5955"),
    ("tiny_shakespeare.txt", "Literature", "Aggregation", "How many times does the speaker label \"ROMEO:\" appear in the text?", "163", "computed"),
    ("tiny_shakespeare.txt", "Literature", "Aggregation", "How many times does the speaker label \"JULIET:\" appear in the text?", "125", "computed"),
    # ---- python_pep8_style_guide.txt
    ("python_pep8_style_guide.txt", "Style Guide", "Multi", "What maximum line length does PEP 8 recommend for code and for docstrings/comments?", "79|72", "Maximum Line Length"),
    ("python_pep8_style_guide.txt", "Style Guide", "Multi", "In what order should imports be grouped?", "standard library|third party|local", "Imports"),
    ("python_pep8_style_guide.txt", "Style Guide", "Yes No", "Should comparisons to singletons like None be done with == ?", "no", "Programming Recommendations"),
    ("python_pep8_style_guide.txt", "Style Guide", "Fact", "How should constants be named according to PEP 8?", "capital letters~uppercase~upper case|underscores", "Constants"),
    # ---- flask_quickstart.txt
    ("flask_quickstart.txt", "Tutorial", "Multi", "On which address and port does the Flask development server run by default?", "127.0.0.1|5000", "A Minimal Application"),
    ("flask_quickstart.txt", "Tutorial", "Fact", "Which command-line option enables debug mode in Flask?", "--debug", "Debug Mode"),
    ("flask_quickstart.txt", "Tutorial", "Yes No", "By default, does a Flask route answer to POST requests?", "no", "HTTP Methods"),
    ("flask_quickstart.txt", "Tutorial", "Fact", "What is required to use sessions in Flask?", "secret key", "Sessions"),
    # ---- python_pep484_type_hints.docx
    ("python_pep484_type_hints.docx", "Specification", "Fact", "What is Optional[T1] equivalent to in PEP 484?", "union|none", "Union types"),
    ("python_pep484_type_hints.docx", "Specification", "Fact", "What syntax does PEP 484 use for callable types?", "callable|returntype~return type", "Callable"),
    ("python_pep484_type_hints.docx", "Specification", "Fact", "What should function bodies in stub files be?", "ellipsis~...", "Stub Files"),
    ("python_pep484_type_hints.docx", "Specification", "Fact", "How is a function parameter without an annotation treated?", "any", "The Any type"),
    # ---- titanic.csv
    ("titanic.csv", "Dataset", "Aggregation", "How many passengers survived in the Titanic dataset?", "342", "computed"),
    ("titanic.csv", "Dataset", "Aggregation", "How many passengers travelled in 3rd class (Pclass = 3)?", "491", "computed"),
    ("titanic.csv", "Dataset", "Calculation", "What percentage of female passengers survived?", "74.2", "computed"),
    ("titanic.csv", "Dataset", "Calculation", "What is the average passenger age (ignoring missing values)?", "29.7", "computed"),
    # ---- tips.csv
    ("tips.csv", "Dataset", "Calculation", "What is the average tip in the tips dataset?", "3.00~2.998~2.99~3.0", "computed"),
    ("tips.csv", "Dataset", "Aggregation", "How many records are for Sunday?", "76", "computed"),
    ("tips.csv", "Dataset", "Table Lookup", "What is the largest total_bill in the dataset?", "50.81", "computed"),
    ("tips.csv", "Dataset", "Unanswerable", "Which waiter served the table with the highest bill?", "", "-"),
    # ---- penguins.csv
    ("penguins.csv", "Dataset", "Aggregation", "How many Gentoo penguins are in the dataset?", "124", "computed"),
    ("penguins.csv", "Dataset", "Calculation", "What is the average body mass of Gentoo penguins?", "5076", "computed"),
    ("penguins.csv", "Dataset", "Table Lookup", "What is the highest body mass recorded and which species is it?", "6300|gentoo", "computed"),
    ("penguins.csv", "Dataset", "Aggregation", "How many penguins were recorded on Torgersen island?", "52", "computed"),
    # ---- car_mpg.csv
    ("car_mpg.csv", "Dataset", "Aggregation", "How many cars have origin japan?", "79", "computed"),
    ("car_mpg.csv", "Dataset", "Table Lookup", "Which car has the highest mpg and what is it?", "mazda glc|46.6", "computed"),
    ("car_mpg.csv", "Dataset", "Table Lookup", "Which car is the heaviest and what is its weight?", "pontiac safari|5140", "computed"),
    ("car_mpg.csv", "Dataset", "Calculation", "What is the average mpg of cars with origin japan?", "30.45", "computed"),
    # ---- airline_passengers.csv
    ("airline_passengers.csv", "Dataset", "Table Lookup", "How many passengers were recorded in July 1960?", "622", "computed"),
    ("airline_passengers.csv", "Dataset", "Table Lookup", "In which month were the fewest passengers recorded and how many?", "1949-11~november 1949|104", "computed"),
    ("airline_passengers.csv", "Dataset", "Calculation", "What is the total number of passengers recorded in 1949?", "1520", "computed"),
    ("airline_passengers.csv", "Dataset", "Table Lookup", "How many passengers were recorded in July 1955?", "364", "computed"),
]

DIFFICULTY = {"Fact": "easy", "Yes No": "easy", "Unanswerable": "medium", "Multi": "medium",
              "Table Lookup": "medium", "Calculation": "hard", "Aggregation": "hard"}