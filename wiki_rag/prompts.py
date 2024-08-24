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
Considerando il contesto fornito:

Contesto da KB: {context}

Contesto dal web: {web_context}

e la domanda posta dall'utente: {query}

genera una risposta dettagliata e pertinente che risponda in modo chiaro e completo alla domanda, tenendo conto del contesto specificato. Assicurati che la risposta sia formulata in un linguaggio comprensibile per l'utente e che includa esempi o spiegazioni aggiuntive se necessario.
Se non conosci la risposta, dì semplicemente che non lo sai, non cercare di inventare una risposta.
"""

UNPERFROMING_PROMPT_IT = """\
Usa i seguenti pezzi di contesto per rispondere alla domanda alla fine.
Se non conosci la risposta, dì semplicemente che non lo sai, non cercare di inventare una risposta.
Dai la priorità alle informazioni più rilevanti e accurate.
Usa un massimo di tre frasi e mantieni la risposta il più concisa, diretta e specifica possibile.

Domanda: {query}

Ecco il contesto:
Contesto da KB: {context}

Contesto dal web: {web_context}

Grazie per aver chiesto!
"""
