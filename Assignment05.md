Github Link:[YorHaaa/atmt_2023: Materials for the first assignment of "Advanced Techniques of Machine Translation" @UZH (Autumn 2023). (github.com)](https://github.com/YorHaaa/atmt_2023)

# 1 Experimenting with Beam Search

Argument:

> Model Select: **The baseline of Assignment 03 (assignments/03/baseline/checkpoints/checkpoint_last.pt)**
>
> batch-size:25
>
> alpha = 0.0 (Not apply the length regularization)

| K             | BLEU Score | Brevity Penalty |
| ------------- | ---------- | --------------- |
| 1             | 15.9       | 1               |
| 3             | 18.0       | 1               |
| 5             | 19.3       | 1               |
| 7             | 19.9       | 1               |
| 10            | 20.8       | 1               |
| 13            | 21.3       | 1               |
| 15            | 21.4       | 1               |
| 17            | 21.7       | 1               |
| 20            | 21.7       | 0.977           |
| 23            | 21.2       | 0.932           |
| 25            | 21.0       | 0.899           |
| greedy_search | 16.8       | 1               |

## Visualize the resulting BLEU scores in an appropriate plot and discuss them in one to two paragraphs of your PDF report. What is the effect of using larger beam sizes on decoding time?

<img src="C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231211215135533.png" alt="image-20231211215135533" style="zoom:50%;" />

> As we can see in the figure above, the BELU score increases between k=(0,17), and then the BELU score starts to decrease. And the Brevity Penalty remains 1 between k = (0,17), and then decreases extremely fast. That means beam search actually improve the performance of the model, but when the beam size is too big, the beam search prefer generating short text (can be seen in the translation result, from k=7, there are a lot of empty sentences), this is because when the generated text is longer, its overall probability will be smaller because it is the product of multiple conditional probabilities. So that we can see the Brevity Penalty decreased very fast when k is getting larger. When I increases the beam size, the time spent increases exponentially.
>
> And from the result of k=1. As the figure shows. The result of beam search is same as the greedy search(Although the empty sentences make the BLEU score of greedy search is a little higher than the beam search). It proves that beam search with k=1 is equal to the greedy search. 
>
> <img src="C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231211210715970.png" alt="image-20231211210715970" style="zoom:50%;" />

# 2 Understanding the Code

## 2.1 QUESTION 1: What is "go_slice" used for and what do its dimensions represent?

> go_slice is a tensor used as the initial input of the decoder. Its dimension is (batch size, 1), which means that each sentence has only one word. 
> The value is the index of  </s>(eos_idx) , which is used to tell the decoder to start. 

## 2.2 QUESTION 2: Why do we keep one top candidate more than the beam size?

> According to the following code :
>
> ````python
> next_word = torch.where(best_candidate == tgt_dict.unk_idx, backoff_candidate, best_candidate)
> log_p = torch.where(best_candidate == tgt_dict.unk_idx, backoff_log_p, best_log_p)
> ````
>
> The reason why we keep one top candidate more than the beam size is that we could avoid the model generating the <UNK> word. When the model predict the <UNK> word have the highest probability (best_candidate == tgt_dict.unk_idx), the model will use the backoff_candidate instead.

## 2.3 QUESTION 3: Why do we add the node with a negative score?

> ![image-20231209192534206](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231209192534206.png)
>
> As the code showing: The beam search use the **Priority Queue** to speed up the process. And we use the log_probability to compute the score `torch.log(torch.softmax(decoder_out, dim=2)`, which means the highest probability has the highest negative value .Because the  head of the Priority Queue have the lowest value, we need to turn the score into negative to get a positive priority to make sure the node with highest probability on the head of the Priority Queue .

## 2.4 QUESTION 4: How are "add" and "add_final" different?What would happen if we did not make this distinction?

> The add function is used when the last word is not EOS and the add_final function is used when the last word is EOS.
>
> If we use the add() method to add an ended candidate sequence, it will cause the sequence to be put into self.nodes instead of self.final, so that the final generated text cannot be correctly obtained, but only Intermediate candidate sequence. And also, if we don't use the add_final() function, we can't ensure all the candidate sequence have the same length, and it will cause missmatching of the dimensions.

## 2.5 QUESTION 5: What happens internally when we prune our beams? How do we know we always maintain the best sequences?

> ![image-20231209194939965](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231209194939965.png)
>
> In the code above,  when we prune our beams, we will first get all the nodes which is ended with EOS (The sequence is finished). And then we get the `self.beam_size-finished` numbers（Because there are *`finished`* numbers of candidate sequences had stored in the `self.final` and removed from the nodes,so we just need the rest nodes） of the candidate sequences of the current search.
>
> After get() function, we could get the nodes with lowest value which means highest probability,then this element will remove from the  queue. So we we always maintain the best sequences.

## 2.6 QUESTION 6: What is the purpose of this for loop?

> First of all, `first_eos = np.where(sent == tgt_dict.eos_idx)[0]` get the index of the first EOS symbols in the final sentences which means the end of the sentences because some sentences may have serval EOS symbols. 
>
> `if len(first_eos) > 0:` means that the EOS symbol exist in the sentences, so we should use `temp.append(sent[:first_eos[0]])` ensure that there is only 1 EOS symbol.
>
> In a word,  the purpose of this loop is to remove the redundant EOS in the sentence.

# 3 UID Decoding

## 3.1 Write a short paragraph about how your implementation (e.g., where you added changes etc.)

According to the formula
$$
Rsquare(y) = \sum_{t=1}^{|y|}ut(yt)^2
$$
The regular value sums the square of probability of each time step. 

1. First, I modified the `get_args` function and add a lambda value in the argument.

![image-20231210125238610](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210125238610.png)

1. I add a new argument `square_regular` in `init` function of BeamSearchNode to compute the current node's regular value.

   ![image-20231210125405312](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210125405312.png)

2. In every time step when adding a new node to the search, I update the value of square regular. (Count the sum of each time step)

   **When creating the node:** the value of square regular is square of current log_p

   ![image-20231210135305877](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210135305877.png)

   **When extending the candidate sequence:**

   ![image-20231210145939269](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210145939269.png)

   > When adding the EOS token to the candidate sequence, remain the value.

3. I modified the eval() function so that I can apply the square regularization in every time step the model generating new word.

   ![image-20231210170643942](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210170643942.png)

   ## 3.2 Analyze and describe the output. (How) does it differ compared to unregularized beam search?

   The output of applying UID Decoding:

   `````Json
   {
    "name": "BLEU",
    "score": 16.5,
    "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
    "verbose_score": "47.9/22.0/11.7/6.0 (BP = 1.000 ratio = 1.164 hyp_len = 4532 ref_len = 3892)",
    "nrefs": "1",
    "case": "mixed",
    "eff": "no",
    "tok": "13a",
    "smooth": "exp",
    "version": "2.3.1"
   }
   `````

   The output of not applying:

   `````Json
   {
    "name": "BLEU",
    "score": 19.3,
    "signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.3.1",
    "verbose_score": "48.3/24.8/14.2/8.2 (BP = 1.000 ratio = 1.212 hyp_len = 4719 ref_len = 3892)",
    "nrefs": "1",
    "case": "mixed",
    "eff": "no",
    "tok": "13a",
    "smooth": "exp",
    "version": "2.3.1"
   }
   `````

   > We can see that the BLEU score increased about 0.6 after applying UID Decoding, and the verbose_score in all n-gram increased a little.
   >
   > And we can see from the result of translation(left file - after applying UID Decoding, right file - beam_search)
   >
   > ![image-20231210165615144](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210165615144.png)
   >
   > ![image-20231210165802561](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210165802561.png)
   >
   > ![image-20231210165853519](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210165853519.png)
   >
   > ![image-20231210170043591](C:\Users\72924\AppData\Roaming\Typora\typora-user-images\image-20231210170043591.png)
   >
   > The result of beam_search after applying UID Decoding seems to be more in line with human language habits and more accurate.

# 4 Investigating the Diversity of Beam Search

Argument:

> k=5 , alpha = 0.7

I choose beam size of 5  because when beam size > 7 , there are a lot of empty sentences in the result. So I don't decide to use the k value which has the highest BLEU score. 

## 4.1 Write a short paragraph about how you implemented the diverse beam search (e.g. where you added changes etc.)

1. I created 2 new file 

    **beam_diverse.py** -Finished the get_n_best function. So that we can get n best candidate sequences after beam search.

   Main idea: `nodes = [(node[0], node[2]) for node in [merged.get() for _ in range(n)]]`

   **translation_beam_diverse.py** -  Finished the diverse beam search

2. Added two arguments

   ```python
   # Add diverse beam search argument
   parser.add_argument('--n-best', default=0, type=int, help='numbers of n best hypotheses in the result')
   parser.add_argument('--gamma', default=0.0, type=float, help='gamma for diverse beam search')
   ```

3. turn the dictionary `all_hyps` into a list, stored the n result of translation.

   ```python
   all_hyps_list = [dict() for _ in range(args.n_best)]
   ```

4. When create number of beam_size next nodes for every current node, I add the diversity penalty to the current log_probability 

   ```python
   # Adjust log_p to reflect rank among siblings (Group the probability by current beam size)
   log_p = log_p[-1] - (j + 1) * args.gamma
   ```

5. When beam search completed, every search could get n best sequences, and then write the result into txt file

   ```python
   for search in searches:
       n_best = search.get_n_best(args.n_best)
       for i in range(len(n_best)):
           n_best_sents.append(n_best[i][1].sequence[1:].cpu())
   ```
