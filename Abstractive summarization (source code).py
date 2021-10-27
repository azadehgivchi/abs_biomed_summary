from owlready2 import *
from statsmodels.compat import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import fitz
default_world.set_backend(filename="pym.sqlite3")
PYM = get_ontology("http://PYM/").load()
CUI = PYM["CUI"]
import PyPDF2
from nltk.corpus import stopwords
import scispacy
import spacy
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from scispacy.umls_linking import UmlsEntityLinker
from collections import defaultdict
from matplotlib import pyplot as plt
import networkx as nx
from rouge import Rouge
from stop_words import get_stop_words
from spacy.lang.en import English
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import csv
from sklearn.metrics.cluster import adjusted_rand_score
import dict_digger
import operator
import nmslib
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from stop_words import safe_get_stop_words
import nltk
import sys
from operator import itemgetter
from itertools import groupby
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
import nltk.data
import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
import re


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

fpp = open("r_abstractive.txt", "w", encoding='utf-8')

start_no = 1# strart index of paper number
end_no = 384 #end index of paper number

dif = end_no - start_no + 1  # number of papers

min_supp = 0.09 # set minimum support
pr7 = 0
pr8 = 0
pr9 = 0
pr11= 0
pr21= 0
pr31= 0
pr4 = 0
pr5 = 0
pr6 = 0

while start_no <= end_no:

    best_score=''

    print('p'+ str(start_no) + ":")
    pr1 = 0
    pr2 = 0
    pr3 = 0
    pr4 = 0
    my_ex = ''  # parameter for exractive summary
    styp =0
    etyp =3

    for m in range(styp, etyp):  # consider 3 type of nlp in biomedical for better result
        if m == 0:
            nlp = spacy.load("en_core_sci_sm")  # small domain
            v = "en_core_sci_sm"
        if m == 1:
            nlp = spacy.load("en_core_sci_lg")  # large domain
            v = "en_core_sci_lg"
        if m == 2:
            nlp = spacy.load("en_core_sci_md")  # medium domain
            v = "en_core_sci_md"
        ax = []
        st = []
        bk = []
        main_doc = []
        stop_p = []
        item = []
        sen_num = []
        # stop = stopwords.words('english')
        bk = []
        ck = []
        tokenized_text = []
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

        f = open("abs_biomed_summary/abstractive_dataset/nlm_papers/nlm_p" + str(start_no) + ".txt", encoding="cp1252")  # read text file of paper

        ax = f.read()
        # delete References ,Abbreviations and Acknowledgements from papers
        ax = ax[:ax.find('References')]
        ax = ax[:ax.find('Abbreviations')]
        ax = ax[:ax.find('Acknowledgements')]

        ax = ax.lower()  # change text to lowercase letter
        ax = ax.replace("\n", " ")  # remove extra character
        ax = ax.replace("\r", " ")  # remove extra character
        ax = ax.replace('vivo', '-vivo')  # delete
        ax = ax.replace('vitro', '-vitro')  # delete
        ax = ax.replace('silico', '-silico')  # delete

        tokenized_text = ax.split()
        # print(' number of words for this paper:', len(tokenized_text))

        fk = nlp(ax)
        bk = list(fk.sents)
        bw = list(fk.ents)
        main_doc = bk
        m = len(bk)

        n_sen = round(0.30 * len(main_doc))  # consider 30% of sentences for summary

        stop_words = set(stopwords.words('english'))
        stop_words1 = get_stop_words('en')
        s = open("f:\stop_words.txt", "r")
        stop_words3 = s.read()
        stop_words3 = stop_words3.split('\n')
        word_tokens = []
        for i in range(0, len(bk)):
            ax = str(bk[i])
            word_tokens = word_tokenize(ax)
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            filtered_sentence = []

            for w in word_tokens:
                if w not in stop_words:
                    if w not in stop_words1:
                        if w not in stop_words3:
                            filtered_sentence.append(w)
            ck.append(' '.join(filtered_sentence))
            filtered_sentence = []
        f.close()
        bk = ck
        dataset = []
        dd = []
        main_doc_split = []
        z = 0
        result = []
        final_re = []
        main_s = []
        st = ''
        # create concept set
        for i in range(0, len(bk)):
            doc1 = '"""' + bk[i] + '"""'
            doc2 = nlp(doc1)
            doc3 = doc2.ents

            if len(doc3) == 0:
                st = ''
                dd.append(st)
            #############################
            for s in range(0, len(doc3)):
                main_s.append(doc3[s])
            main_doc_split.append(main_s)
            main_s = []
            ###########################
            for i in range(0, len(doc3)):
                st = str(doc3[i]).lower()
                dd.append(st)
                # print(st)
                ##########################
                flag = 0
                co = ''
                concept = CUI.search('"' + st + '"')  # find concepts of the tokens from NLM
                if concept == []:  # Put entity as its concept if  thers isn't  concept for it
                    dd.append(st)
                for j in range(0, len(concept)):
                    co = str(((concept[j].label)[0])).split()

                    if concept[j].name[
                       0:3] == 'C00':  # concat all concepts of entity if code of its name starts with 'coo'
                        xc = ' '.join(co)
                        dd.append(xc.lower())
                    else:
                        dd.append(st)

            dd = list(dict.fromkeys(dd))
            dataset.append(dd)
            z = len(dd) + z
            dd = []
        final_re = dataset
        # -----------------------------------
        itt = []
        itt1 = []
        se = []
        itt3 = []
        itt4 = []
        ###########################################
        sys.setrecursionlimit(999999990)

        # fpgrowth algorithem
        te = TransactionEncoder()
        te_ary = te.fit(dataset).transform(dataset)
        df = pd.DataFrame(te_ary, columns=te.columns_)
        itemm = fpgrowth(df, min_support=min_supp, use_colnames=True)
        item_def = pd.DataFrame(itemm, columns=['support', 'itemsets'])
        itemsets = item_def['itemsets'].to_list()
        itemprc = item_def['support'].to_list()

        stype =0
        etype =4

        for o in range(stype, etype):

            for i in range(0, len(itemsets)):
                itt.append(list(itemsets[i]))

            for j in range(0, len(itemprc)):
                itt1.append(itemprc[j])

            if o == 0:

                for i in range(0, len(itt)):
                    cc = [1]  # Frequent items of length 1

                    if len(itt[i]) in cc:
                        se.append(i)

            if o == 1:
                for i in range(0, len(itt)):
                    cc = [1, 2]  # Frequent items of length 1,2

                    if len(itt[i]) in cc:
                        se.append(i)

            if o == 2:
                for i in range(0, len(itt)):
                    cc = [1, 2, 3]  # Frequent items of length 1,2,3

                    if len(itt[i]) in cc:
                        se.append(i)

            if o == 3:
                for i in range(0, len(itt)):
                    cc = [1, 2, 3, 4]  # Frequent items of length 1,2,3,4

                    if len(itt[i]) in cc:
                        se.append(i)
            # ----------------------------------------------------------------------------
            for j in range(0, len(se)):
                itt3.append(itt[se[j]])

            for j in range(0, len(se)):
                itt4.append(itt1[se[j]])
            itt = itt3
            itt1 = itt4

            ef = []
            f1 = []
            st = []

            ef3 = []
            ef4 = []
            ef5 = []
            ef6 = []
            ef9 = []
            counter = 0
            efw = []
            efn = []
            # -calculate weight of sentences

            for i in range(0, len(dataset)):

                f1 = dataset[i]
                l1 = len(f1)

                counter = 0
                nu = 0
                for t in range(0, len(itt)):

                    if len(itt[t]) == 1:
                        x = (itt[t])[0]
                        y = itt1[t]

                        if str(x) in str(f1):
                            counter = counter + y
                            nu = nu + 1
                    else:
                        st = itt[t]
                        flag = 0
                        for g in range(0, len(st)):
                            if str(st[g]) not in str(f1):
                                flag = 1
                        if flag == 0:
                            ef.append(itt[t])
                            counter = counter + y
                            nu = nu + 1
                ef3.append(i)
                ef9.append(counter)
                efw.append(counter)
                efn.append(nu)
                if l1 == 0:
                    ef3.append(0)
                    ef5.append(0)

                else:
                    ef3.append(nu)
                    ef5.append(nu)

                ef5.append(counter)
                ef3.append(counter)

                ef4.append(ef3)
                ef6.append(ef5)
                ef5 = []
                ef3 = []
            # -create matrix for mst
            cs = []
            cs1 = []
            tel = 0.0
            main_sen = []
            main_sen1 = []
            # Calculate the weight of the edges of the graph
            for j in range(0, len(dataset)):
                f1 = dataset[j]
                l1 = len(f1)
                inter = 0
                c7 = []
                for k in range(0, len(dataset)):
                    f2 = dataset[k]
                    l2 = len(f2)
                    inter = 0
                    c7 = []
                    for p in range(0, len(f1)):
                        for q in range(0, len(f2)):
                            if str(f1[p]).strip() == str(f2[q]).strip():
                                if f1[p] not in c7:
                                    c7.append(f1[p])
                                    inter = inter + 1
                    if (efn[k] + efn[j]) - inter > 0:

                        e = (len(intersection(f1, f2)) / ((l1 + l2))) + efw[k] + efw[j]

                    else:
                        e = 0
                    cs1.append(e)
                    if e > tel:
                        main_sen1.append(j)
                        main_sen1.append(k)
                main_sen.append(main_sen1)
                cs.append(cs1)
                cs1 = []
                main_sen1 = []

            # get maximum weight in graph
            maxx = cs[0][0]
            for m in range(0, len(cs)):
                for t in range(m, len(cs)):
                    if cs[m][t] > maxx:
                        maxx = cs[m][t]

            for i1 in range(0, len(cs)):
                for j1 in range(0, len(cs)):
                    cs[i1][j1] = (cs[i1][j1] * -1) + maxx + 1

            # finding shortest path by MST algorithem

            cs = np.array(cs, dtype=np.float32)
            cd = csr_matrix(cs)

            tcsr = minimum_spanning_tree(cd)
            tcsr = tcsr.toarray().astype(float)
            n_components = connected_components(csgraph=cs, directed=False, return_labels=False)
            sen = []
            sen1 = []
            # anlyse shortest path and finding sentences that are in this path.
            # the shortest path shows in the matrix that only the value of shortest path is not zero
            # the Coordinates of shotrst path matrix are the sentence numbers of text. so, this sentence numbers are in sen1 list

            for i in range(0, len(tcsr)):
                x = tcsr[i]
                for j in range(0, len(tcsr)):
                    if x[j] != 0.0 and i != j:
                        sen1.append(j)
            c6 = []
            c6 = sen1
            c57 = []
            c56 = []
            c60 = c6
            c56 = []

            c58 = []
            ll = len(c6)

            # Get the number of repetitions Of node or degree of nodes

            for s in range(0, len(c6)):
                cc = c6[s]
                c56.append(cc)
                v = 0
                for k in range(0, len(c6)):
                    if c6[k] == cc:
                        v = v + 1
                c56.append(v)
                c57.append(c56)
                c56 = []

            c57 = list(k for k, _ in itertools.groupby(c57))
            c57 = sorted(c57, key=operator.itemgetter(1))  # desending sort according to the degree of nodes(sentences)

            c6 = []
            # sort according to the locations of these sentences on main text
            for t in range(0, len(c57)):
                zz = c57[t]
                c6.append(zz[0])

            c = 0
            bp = 0
            sen_num = []

            if len(c6) < n_sen: n_sen = len(c6)

            for r in range(0, n_sen):
                qq = c6[r]
                sen_num.append(qq)

            sen_num.sort()
            my_extractive = ''

            # create extractive summary by concat thses sentences

            for rq in range(0, round(len(sen_num))):
                my_extractive = my_extractive + str(main_doc[sen_num[rq]])

            my_extractive = my_extractive.lower()
            f = open("abs_biomed_summary/abstractive_dataset/title of papers/nlm_abs" + str(start_no) + ".txt", encoding="cp1252")

            main_abstarctive1 = f.read()  # read the title of paper
            main_abstarctive1 = main_abstarctive1.replace("\n", " ")  # remove extra character
            main_abstarctive1 = main_abstarctive1.replace("\r", " ")  # remove extra character

            abs = ''
            out1 = ''
            output = ''

            my_w = len(my_extractive.split())
            my_s = nlp(my_extractive)
            my_s1 = len(list(my_s.sents))
            my_s2 = list(my_s.sents)

            if my_w <= 500:  # set splite number according to the lenght of extractive summary
                vb = 200
            if my_w > 1000:
                vb = 800
            if my_w > 500 and my_w <= 1000:
                vb = 700

            if my_w >= 500:
                j = 0
                flag = 0
                while j < my_s1:
                    i = j
                    flag = 0
                    while flag == 0 and i < my_s1:
                        if len(abs.split()) <= vb:
                            abs = abs + str(my_s2[i])
                            i = i + 1
                        else:
                            flag = 1
                    j = i + 1

                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                device = torch.device('cpu')

                preprocess_text = abs.strip().replace("\n", "")
                t5_prepared_Text = "summarize: " + preprocess_text
                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

                # summmarize

                summary_ids = model.generate(
                    tokenized_text,
                    max_length=150,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                out1 = output
                out1 = out1 + abs

                abs = ''
                from_tf = True
                xx = len(out1.split())
                rouge = Rouge()
                ab_scores = rouge.get_scores(main_abstarctive1, out1)

            else: # if lenght of extractive summary  is less than 500 words
                model = T5ForConditionalGeneration.from_pretrained('t5-small')
                tokenizer = T5Tokenizer.from_pretrained('t5-small')
                device = torch.device('cpu')
                text = my_extractive
                preprocess_text = text.strip().replace("\n", "")
                t5_prepared_Text = "summarize: " + preprocess_text
                tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
                summary_ids = model.generate(
                    tokenized_text,
                    max_length=150,
                    num_beams=2,
                    repetition_penalty=2.5,
                    length_penalty=1.0,
                    early_stopping=True)

                output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            rouge = Rouge()
            ab_scores = rouge.get_scores(main_abstarctive1, output)

            #print("\n\nSummarized text: \n", output)

            pr1 = float(dict_digger.dig(ab_scores[0], 'rouge-1', 'p'))
            pr2 = float(dict_digger.dig(ab_scores[0], 'rouge-2', 'p'))
            pr3 = float(dict_digger.dig(ab_scores[0], 'rouge-l', 'p'))
            if best_score=='':
                best_score=ab_scores
                my_ab=output
                my_exe = my_extractive
                pr4 = pr1
                pr5 = pr2
                pr6 = pr3

            if pr1 > pr4:
                best_score = ab_scores
                pr4 = pr1
                pr5 = pr2
                pr6 = pr3
                my_ab = output
                my_exe=my_extractive

    print(best_score)
    print(my_ab)
    fpp.write('P' +str(start_no) +":"+'\n'+'\n')
    fpp.write('Title:' + '\n')
    fpp.write(str(main_abstarctive1) + '\n')
    fpp.write('*****************************************'+ '\n')
    fpp.write('Extractive summary:' + '\n')
    fpp.write(str(my_extractive) + '\n')
    fpp.write('*****************************************'+ '\n')
    fpp.write('Abstractive summary:' + '\n')
    fpp.write(str(my_ab) + '\n')
    fpp.write('*****************************************'+ '\n')
    fpp.write('number of words= ' + str(len(my_ab.split())) + '\n')
    fpp.write(str(best_score) + '\n')
    fpp.write('-----------------------------------------------------------------------------------------------------------------------------------------------'+'\n')
    print('---------------------------------------------------------------------------------------------------------------------------------------------------------')
    pr7 = float(dict_digger.dig(ab_scores[0], 'rouge-1', 'p'))
    pr8 = float(dict_digger.dig(ab_scores[0], 'rouge-1', 'r'))
    pr9 = float(dict_digger.dig(ab_scores[0], 'rouge-1', 'f'))

    pr11 = pr11 + pr4
    pr21 = pr21 + pr5
    pr31 = pr31+ pr6

    pr7=0
    pr8=0
    pr9=0
    start_no = start_no + 1

fpp.write('ROUG1= ' + str(pr11 / dif) + '\n')
fpp.write('ROUG2= ' + str(pr21 / dif) + '\n')
fpp.write('ROUG3= ' + str(pr31 / dif) + '\n')

print('rouge1=', pr11/ dif)
print('rouge2=', pr21 / dif)
print('rougel=', pr31 / dif)





