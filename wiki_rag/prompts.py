"""File with all the prompts for the wiki_rag model.
"""

ANSWER_QUESTION_TEMPLATE_EN = """\

Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

Question: {query}

Here is the context
Context from KB: {context}

Context from the web: {web_context}
"""

ANSWER_QUESTION_TEMPLATE_IT = """\
Usa i seguenti pezzi di contesto per rispondere alla domanda alla fine.
Se non conosci la risposta, dì semplicemente che non lo sai, non cercare di inventare una risposta.
Usa un massimo di tre frasi e mantieni la risposta il più concisa possibile.

Domanda: {query}

Ecco il contesto:
Contesto da KB: {context}

Contesto dal web: {web_context}
"""
