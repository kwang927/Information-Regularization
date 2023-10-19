# Prompt for regularization on query generation
The specific prompts for each regularization are in the pickle file. We also give sample prompts for each prompt we provided in this folder below.

## Table of Contents
* [Format](#file-format)
* [DORIS-MAE](#doris-mae)
    * [Instruction regularization prompt](#doris-mae-instruction-regularization-prompt)
        * [Query regularization on Instruction regularization prompt](#doris-mae-query-regularization-on-instruction-regularization-prompt)
    * [Document regularization 40% prompt](#doris-mae-document-regularization-40-prompt)
        * [Query Regularization on Document regularization 40% prompt](#doris-mae-query-regularization-on-document-regularization-40-prompt)
    * [Document regularization 80% prompt](#doris-mae-document-regularization-80-prompt)
        * [Query Regularization on Document regularization 80% prompt](#doris-mae-query-regularization-on-document-regularization-80-prompt)
    * [Get keywords prompt](#doris-mae-get-keywords-prompt)
    * [Promptagator prompt example](#doris-mae-promptagator-prompt)
* [ArguAna](#arguana)
    * [Instruction regularization prompt](#arguana-instruction-prompt)
        * [Query regularization on Instruction regularization prompt](#arguana-query-regularization-on-instruction-regularization-prompt)
    * [Document regularization 40% prompt](#arguana-document-regularization-40-prompt)
        * [Query Regularization on Document regularization 40% prompt](#arguana-query-regularization-on-document-regularization-40-prompt)
    * [Document regularization 80% prompt](#arguana-document-regularization-80-prompt)
        * [Query Regularization on Document regularization 80% prompt](#arguana-query-regularization-on-document-regularization-80-prompt)
    * [Get keywords prompt](#arguana-get-keywords-prompt)
    * [Promptagator prompt example](#arguana-promptagator-prompt)
* [WhatsThatBook](#whatsthatbook)
    * [Instruction regularization prompt](#wtb-instruction-regularization-prompt)
        * [Query regularization on Instruction regularization prompt](#wtb-query-regularization-on-instruction-regularization-prompt)
    * [Document regularization 40% prompt](#wtb-document-regularization-40-prompt)
        * [Query Regularization on Document regularization 40% prompt](#wtb-query-regularization-on-document-regularization-40-prompt)
    * [Document regularization 80% prompt](#wtb-document-regularization-80-prompt)
        * [Query Regularization on Document regularization 80% prompt](#wtb-query-regularization-on-document-regularization-80-prompt)
    * [Get keywords prompt](#wtb-get-keywords-prompt)
    * [Promptagator prompt example](#wtb-promptagator-prompt)

## File Format
The files are in form `{dataset_name}_{type}_prompt.pickle`.
The types include:
- `Ireg`: Instruction regularization
    - `Qreg_Ireg`: Query regularization on Instruction regularization
- `Dreg_{p}%`: Document p% regularization
    - `Qreg_Dreg_{p}%`: Query regularization on Document p% regularization
- `promptagator`: [Promptagator style](https://iclr.cc/virtual/2023/poster/10937)  


## DORIS-MAE
### DORIS-MAE Instruction regularization prompt
    I want you to transform computer science abstracts into queries in a particular manner. Here is one example of the transformation:
    {Example_abstract}
    ->
    Example Query: {Example_query}

    I am going to present you with a new abstract. You should transform it into a new query, while satisfying the following requirements:
    1. The query should, broadly speaking, be addressed by the abstract.
    2. The query should not contain the distinctive keywords contained in an abstract. Put another way, simple keyword matching *should not* be sufficient to retrieve the abstract given the query. When you encounter distinctive keywords, think of alternative paraphrases. This is very important.
    3. The query should describe a natural set of questions or goals that a scientific researcher could have.

    Abstract:
    {Abstract for new query}

    Before giving the final query, first think step by step, to make sure that you are satisfying the following constraints.
    1. Identify the important questions that are addressed by the abstract.
    2. Paraphrase each important question. In order to decide whether a key phrase has been sufficiently reworked, think about whether a naive search engine would be able to find the abstract given the phrase. Highly specialized terms -- terms that are introduced by the abstract which are likely to be distinctive to only a few papers -- should not be included. However, you *should not* paraphrase common technical terms which occur in many papers.
    3. Consolidate the questions into a natural query -- not all questions need to be included, and you can combine/synthesize questions to make things more natural.
    4. Write the query.
    Do steps 1-4 above incrementally

### DORIS-MAE Query regularization on Instruction regularization prompt
    Given a complex query, extract 4 disjoint separate problem statements from it.
    Query:

### DORIS-MAE Document regularization 40% prompt
    Here is a redacted scientific abstract: "{Redacted_abstract}".

    Here is the example query based on other abstract: "{Example_query}".

    Follow the style of the example query, write a new **query** based on the provided abstract.

    Only output the new query (strictly more than 125 words). The query should not contain "_".  Note that the query should mimic the style of the example query and base on the redacted query.

### DORIS-MAE Query regularization on Document regularization 40% prompt
    Given a complex query, extract 3 disjoint separate problem statements from it.
    Query: {query}

### DORIS-MAE Document regularization 80% prompt
    Here is a heavily redacted scientific abstract: "{Redacted_abstract}".

    Here is the example query based on other abstract: "{Example_query}".

    Follow the style of the example query, write a new **query** based on the provided abstract.

    Only output the new query (strictly more than 125 words). The query should not contain "_".  Note that the query should mimic the style of the example query and base on the redacted query.

### DORIS-MAE Query regularization on Document regularization 80% prompt
    Given a complex query, extract 3 disjoint separate problem statements from it.
    Query: {query}

### DORIS-MAE Get keywords prompt
    Here is a paragraph: {Abstract}. Give me **all** the important words that represent salient ideas in the paragraph. Output the words in a python list format.

### DORIS-MAE Promptagator prompt
    Passage_0: {Abstract_0}

    Query_0: {Query_0}

    Passage_1: {Abstract_1}

    Query_1: {Query_1}

    Passage_2: {Abstract_2}

    Query_2: {Query_2}

    Passage_3: {Abstract_3}

    Query_3: {Query_3}

    Passage_4: {Abstract_4}

    Query_4: {Query_4}

    Passage_5: {Abstract_5}

    Query_5: {Query_5}

    Passage_6: {Abstract_6}

    Query_6: {Query_6}

    Passage_7: {Abstract_7}

    Query_7: {Query_7}

    Passage_8: {Abstract for new query}

    Write Query_8

## ArguAna
### ArguAna Instruction prompt
    I want you to transform the argument to counterargument in a particular manner. Here is one example of the transformation:
    
    {Example_argument} ->

    {Example_counterargument}

    As a debater, I am going to present you with a new argument. You should transform the argument to a counterargument while satisfying the following requirements:
    1. The counterargument takes on the aspects of the topic invoked by the argument, while adding a new perspective to its conclusion and/or premises, conveying the opposite stance.
    2. The counterargument should not refute the argument in any sense. Instead, the counterargument should be an independent argument that conveys another point of view. Think of this as if you are the first speaker in a debate and you should primarily focus on introducing and stating your point of view. You should present your stand and arguments on the given topic. You should not refute the opponent's ideas as these arguments have not yet been presented.

    Argument: {Argument for new counterargument}

### ArguAna Query regularization on Instruction regularization prompt
    Given an argument, summarize it down to a shorter argument with at most 50 words.
    Argument: {argument}

### ArguAna Document regularization 40% prompt
    Given a redacted argument that has multiple positions it supports, determine one of the positions that it supports, and generate a one paragraph counterargument that has a new perspective on it. The counterargument should be an independent argument. The counter argument *Must Not* have the word argument in it. Output the counter argument after the key word "Counter:"
        
    {redacted_argument}

### ArguAna Query regularization on Document regularization 40% prompt
    Given an argument, summarize it down to a shorter argument with at most 50 words.
    Argument: {query}


### ArguAna Document regularization 80% prompt
    Given a redacted argument that has multiple positions it supports, determine one of the positions that it supports, and generate a one paragraph counterargument that has a new perspective on it. The counterargument should be an independent argument. The counter argument *Must Not* have the word argument in it. Output the counter argument after the key word "Counter:"
        
    {redacted_argument}

### ArguAna Query regularization on Document regularization 80% prompt
    Given an argument, summarize it down to a shorter argument with at most 50 words.
    Argument:  {argument}

### ArguAna Get keywords prompt
    Here is an argument: {argument}
    Give me **all** the important words that represent salient ideas in the argument. Output the words in a python list format.

### ArguAna promptagator prompt
    Argument_0:{argumenet_0}

    Counterargument_0:{counterargumenet_0}

    Argument_1:{argumenet_1}

    Counterargument_1:{counterargumenet_1}

    Argument_2:{argumenet_2}

    Counterargument_2:{counterargumenet_2}

    Argument_3:{argumenet_3}

    Counterargument_3:{counterargumenet_3}

    Argument_4:{argumenet_4}

    Counterargument_4:{counterargumenet_4}

    Argument_5:{argumenet_5}

    Counterargument_5:{counterargumenet_5}

    Argument_6:{argumenet_6}

    Counterargument_6:{counterargumenet_6}

    Argument_7:{argumenet_7}

    Counterargument_7:{counterargumenet_7}

    Argument_8:{argumenet for new counterargument}

    Write Counterargument_8


## WhatsThatBook
### wtb Instruction regularization prompt
    I want you to transform the description of a book into a tip of the tongue query in a particular manner. Here is one example of the transformation:

    Example Description:
    {Exmaple_description}

    Example Query:
    {Example_query}

    I am going to present you with a new book description. You should transform it into a new query, while satisfying the following requirements:
    1. The query should, broadly speaking, be addressed by the description.
    2. The query should not contain the distinctive keywords contained in a description. Put another way, simple keyword matching *should not* be sufficient to retrieve the description given the query. When you encounter distinctive keywords, think of alternative paraphrases. This is very important.
    3. The query should be in tip-of-the-tongue format. Think of a situation where a user wants to find a book that they have previously read. The user may be uncertain about identifying details and may rely on creative strategies for describing the information they want to retrieve. These strategies include text that describes content elements (e.g., book characters or events), information beyond the document text (e.g., descriptions of book covers), or personal context (e.g., when they read a book).
    4. The query's length should try to match the new book description length. 

    New book description:
    {Description for new query}

    Before giving the final query, first think step by step, to make sure that you are satisfying the following constraints.
    1. Identify the important features in the description.
    2. Paraphrase each important feature. In order to decide whether a key phrase has been sufficiently reworked, think about whether a naive search engine would be able to find the abstract given the phrase. Highly specific terms should not be included.
    3. Consolidate the features into a natural query -- not all features need to be included, and you can combine/synthesize features to make things more natural.
    4. Ensure your query matches the length of the new book description. Write *only* the query. 
    Do steps 1-4 above incrementally

### wtb Query regularization on Instruction regularization prompt
    Given a query, summarize it down to a shorter query with at most 50 words.
    Query: {query}

### wtb Document regularization 40% prompt
    Here is a redacted book description: "{redacted_description}"

    Here is an example tip-of-tongue query based on an example book description: "{Example_query}".

    Follow the style of the example tip-of-tongue query, write a new tip-of-tongue **query** based on the provided book description.

    Only output the new query (strictly more than 125 words) in one paragraph. The query should not contain "_".  Note that the query should mimic the style of the example query and entirely base on the redacted book description.

### wtb Query regularization on Document regularization 40% prompt
    Given a query, summarize it down to a shorter query with at most 50 words.
    Query: {query}

### wtb Document regularization 80% prompt
    Here is a heavily redacted book description: "{redacted_description}".

    Here is an example tip-of-tongue query based on an example book description: "{Example_query}".

    Follow the style of the example tip-of-tongue query, write a new tip-of-tongue **query** based on the provided book description.

    Only output the new query (strictly more than 125 words) in one paragraph. The query should not contain "_".  Note that the query should mimic the style of the example query and entirely base on the redacted book description.

### wtb Query regularization on Document regularization 80% prompt
    Given a query, summarize it down to a shorter query with at most 50 words.
    Query: {query}

### wtb Get keywords prompt
    Here is a description of a book: {description}
    Give me **all** the important words that represent salient ideas in the description. Output the words in a python list format.

### wtb promptagator prompt
    Description_1:
    {Description_1}
    Query_1:
    {Query_1}

    Description_2:
    {Description_2}
    Query_2:
    {Query_2}

    Description_3:
    {Description_3}
    Query_3:
    {Query_3}

    Description_4:
    {Description_4}
    Query_4:
    {Query_4}

    Description_5:
    {Description_5}
    Query_5:
    {Query_5}

    Description_6:
    {Description_6}
    Query_6:
    {Query_6}

    Description_7:
    {Description_7}
    Query_7:
    {Query_7}

    Description_8:
    {Description_8}
    Query_8:
    {Query_8}

    Description_9:
    {Description for new query}
    Write Query_9 according to Description_9
