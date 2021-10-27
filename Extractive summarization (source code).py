from owlready2 import *
from statsmodels.compat import scipy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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

fpp = open("extractive.txt", "w", encoding='utf-8')

start_no = 1  # strart index of paper number
end_no = 400 # end index of paper number

dif = end_no - start_no + 1  # number of papers

min_supp = 0.09  # set minimum support
pr7 = 0
pr8 = 0
pr9 = 0

while start_no <= end_no:
    pr1 = 0
    pr2 = 0
    pr3 = 0
    pr4 = 0
    pr5 = 0
    pr6 = 0

    my_ex = ''  # output parameter for exractive summary
    styp = 0
    etyp = 3

    for m in range(styp, etyp):  # consider 3 type of nlp in biomedical for better result
        if m == 0:
            nlp = spacy.load("en_core_sci_sm")  # small domain
            v = "en_core_sci_sm"
        if m == 1:
            nlp = spacy.load("en_core_sci_md")  # medium domain
            v = "en_core_sci_md"

        if m == 2:
            nlp = spacy.load("en_core_sci_lg")  # large domain
            v = "en_core_sci_lg"

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

        f = open("extractive_summary/papers/p" + str(start_no) + ".txt", encoding="utf-8")  # for extractive summary

        ax = f.read()
        # delete References ,Abbreviations and Acknowledgements from papers
        ax = ax[:ax.find('References')]
        ax = ax[:ax.find('Abbreviations')]
        ax = ax[:ax.find('Acknowledgements')]

        ax = ax.lower()  # change text to lowercase letter
        ax = ax.replace("\n", " ")  # remove extra character
        ax = ax.replace("\r", " ")  # remove extra character
        ax = ax.replace('vivo', '-vivo') # these words (vivo, vitro and silico) have problem in python.so, i changed them
        ax = ax.replace('vitro', '-vitro')
        ax = ax.replace('silico', '-silico')

        fb = open("extractive_summary/abstract of papers/ab" + str(start_no) + ".txt", encoding="utf-8") # for extractive summary
        main_abstarctive = fb.read()

        tokenized_text = ax.split()
        fk = nlp(ax)
        bk = list(fk.sents)
        bw = list(fk.ents)
        main_doc = bk
        m = len(bk)

        n_sen = round(0.30 * len(main_doc))  # consider 30% of sentences for extractive summary

        stop_words = set(stopwords.words('english'))
        stop_words1 = get_stop_words('en')
        s = open("stop_words.txt", "r")
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

        stype = 0
        etype = 4

        for o in range(stype, etype): # test ferequent concepts with lengh 1- 1,2-  1,2,3- 1,2,3,4

            for i in range(0, len(itemsets)):
                itt.append(list(itemsets[i]))

            for j in range(0, len(itemprc)):
                itt1.append(itemprc[j])

            if o == 0:

                for i in range(0, len(itt)):
                    cc = [1]  # Frequent concepts of length 1

                    if len(itt[i]) in cc:
                        se.append(i)

            if o == 1:
                for i in range(0, len(itt)):
                    cc = [1, 2]  # Frequent concepts of length 1,2

                    if len(itt[i]) in cc:
                        se.append(i)

            if o == 2:
                for i in range(0, len(itt)):
                    cc = [1, 2, 3]  # Frequent concepts of length 1,2,3

                    if len(itt[i]) in cc:
                        se.append(i)
            if o == 3:
                for i in range(0, len(itt)):
                    cc = [1, 2, 3, 4]  # Frequent concepts of length 1,2,3,4

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

            # Calculate the weight of edges in graph
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
            # the shortest path shows in the matrix that only the locations of shortest path is not zero
            # the x,y of shotrst path matrix are the sentence numbers of text. so, this sentence numbers are in sen1 list

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

            main_abstarctive = main_abstarctive.lower()
            my_extractive = my_extractive.lower()

            rouge = Rouge()
            if my_extractive != '':
                scores = rouge.get_scores(main_abstarctive, my_extractive)

        if pr1 > pr4:
            best_score = scores
            pr4 = pr1
            pr5 = pr2
            pr6 = pr3
            my_ex = my_extractive
    print('p' + str(start_no) + ":")
    print('the best',best_score)
    print('--------------------------------------------------------------------------------------------')

    pr7 = pr7 + pr4
    pr8 = pr8 + pr5
    pr9 = pr9 + pr6

    my_w = len(my_ex.split())  # number of extractive summary words
    my_s = nlp(my_ex)
    my_s1 = len(list(my_s.sents))  # number of sentences of extractive summary
    my_s2 = list(my_s.sents)  # sentences of extractive summary

    fpp.write('p' + str(start_no) + ':' + '\n')
    fpp.write('Extractive Summary:' + '\n')
    fpp.write(str(my_ex) + '\n')
    fpp.write('number of words= ' + str(my_w) + '\n')
    fpp.write(str(best_score) + '\n')
    fpp.write('-----------------------------------------------------------------------------------------------------------------------------------' + '\n')


    pr7 = float(dict_digger.dig(best_score[0], 'rouge-1', 'p')) + pr7
    pr8 = float(dict_digger.dig(best_score[0], 'rouge-1', 'r')) + pr8
    pr9 = float(dict_digger.dig(best_score[0], 'rouge-1', 'f')) + pr9

    start_no = start_no + 1

fpp.write('ROUG1= ' + str(pr7 / dif) + '\n')
fpp.write('ROUG2= ' + str(pr8 / dif) + '\n')
fpp.write('ROUG3= ' + str(pr9 / dif) + '\n')







