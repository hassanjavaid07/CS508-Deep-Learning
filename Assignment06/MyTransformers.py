"""
###<u> **DEEP LEARNING PROGRAMMING ASSIGNMENT # 6** </u>
* **NAME = HASSAN JAVAID**
* **ROLL NO. = MSCS23001**
* **TASK01 = Implementation and Training of Transformer Model**
* **ALGORITHM used: Transformer Model and Attention schemes**
"""



"""
Implements a Transformer in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""

import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from math import sin, cos, floor, ceil
    


def hello_transformers():
    print("Hello from transformers.py!")


def generate_token_dict(vocab):
    """
    The function creates a hash map from the elements in the vocabulary to
    a unique positive integer value.

    args:
        vocab: This is a 1D list of strings containing all the items in the vocab

    Returns:
        token_dict: a python dictionary with key as the string item in the vocab
            and value as a unique integer value
    """
    token_dict = {}
    
    ##############################################################################
    # TODO: Use this function to assign a unique whole number element to each    #
    # element present in the vocab list. To do this, map the first element in the#
    # vocab to 0 and the last element in the vocab to len(vocab) - 1, and the    #
    # elements in between as consecutive numbers.                                #
    ##############################################################################
    
    for idx, token in enumerate(vocab):
        token_dict[token] = idx
    
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    
    return token_dict


def prepocess_input_sequence(input_str: str, token_dict: dict, spc_tokens: list) -> list:
    """
    The goal of this function is to convert an input string into a list of positive
    integers that will enable us to process the string using neural nets further. We
    will use the dictionary made in the previous function to map the elements in the
    string to a unique value. Keep in mind that we assign a value for each integer
    present in the input sequence. For example, for a number present in the input
    sequence "33", you should break it down to a list of digits,
    ['0', '3'] and assign it to a corresponding value in the token_dict.

    args:
        input_str: A single string in the input data
                 e.g.: "BOS POSITIVE 0333 add POSITIVE 0696 EOS"

        token_dict: The token dictionary having key as elements in the string and
            value as a unique positive integer. This is generated using
            generate_token_dict function

        spc_tokens: The special tokens apart from digits.
    Returns:
        out_tokens: a list of integers corresponding to the input string
    """
    out_tokens = []
    ##############################################################################
    # TODO: for each number present in the input sequence, break it down into a
    # list of digits and use this list of digits to assign an appropriate value
    # from token_dict. For special tokens present in the input string, assign an
    # appropriate value for the complete token.
    ##############################################################################
    
    input_elements = input_str.split()

    for element in input_elements:
        if element in spc_tokens:
            out_tokens.append(token_dict[element])
        else:
            for char in element:
                out_tokens.append(token_dict[char])
    
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return out_tokens


def scaled_dot_product_two_loop_single(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embedding dimension

        key: a Tensor of shape (K, M) where K is the sequence length and M is the
            sequence embedding dimension

        value: a Tensor of shape (K, M) where K is the sequence length and M is
            the sequence embedding dimension

    Returns
        out: a tensor of shape (K, M) which is the output of self-attention from
        the function
    """
    K, M = query.shape

    out = torch.zeros_like(query)

    ###############################################################################
    # TODO: Implement this function using exactly two for loops. For each of the  #
    # K queries, compute its dot product with each of the K keys. The scalar      #
    # output of the dot product will the be scaled by dividing it with the sqrt(M)#
    # Once we get all the K scaled weights corresponding to a query, we apply a   #
    # softmax function on them and use the value matrix to compute the weighted   #
    # sum of values using the matrix-vector product. This single vector computed  #
    # using weighted sum becomes an output to the Kth query vector                #
    ###############################################################################
    
    scaling_factor = torch.sqrt(torch.tensor(M, dtype=torch.float32))
    
    for i in range(K):
        scores = torch.zeros(K)
        for j in range(K):
            scores[j] = torch.dot(query[i], key[j])
        
        scores /= scaling_factor
        
        attention_weights = F.softmax(scores, dim=0)
        
        out[i] = torch.matmul(attention_weights, value) # weighted sum of values
    
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    
    return out



def scaled_dot_product_two_loop_batch(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    """
    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. Follow the
    description in TODO for implementation.

    args:
        query: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embedding dimension

        key: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embedding dimension

        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embedding dimension

    Returns:
        out: a tensor of shape (N, K, M) that contains the weighted sum of values
    """
    N, K, M = query.shape
    
    out = torch.zeros_like(query)
    
    ###############################################################################
    # TODO: This function is extending self_attention_two_loop_single for a batch #
    # of N. Implement this function using exactly two for loops. For each N       #
    # we have a query, key and value. The final output is the weighted sum of     #
    # values of these N queries and keys. The weight here is computed using scaled#
    # dot product  between each of the K queries and key. The scaling value here  #
    # is sqrt(M). For each of the N sequences, compute the softmaxed weights and  #
    # use them to compute weighted average of value matrix.                       #
    # Hint: look at torch.bmm                                                     #
    ###############################################################################

    scaling_factor = torch.sqrt(torch.tensor(M, dtype=torch.float32))

    for i in range(N):
        # Looping over batch elements
        for j in range(K):
            scores = torch.zeros(K)
            for k in range(K):
                scores[k] = torch.dot(query[i, j], key[i, k])
            
            scores /= scaling_factor
            
            attention_weights = F.softmax(scores, dim=0)
            
            out[i, j] = torch.matmul(attention_weights, value[i])

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    
    return out



def scaled_dot_product_no_loop_batch(
    query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
) -> Tensor:
    """

    The function performs a fundamental block for attention mechanism, the scaled
    dot product. We map the input query, key, and value to the output. It uses
    Matrix-matrix multiplication to find the scaled weights and then matrix-matrix
    multiplication to find the final output.

    args:
        query: a Tensor of shape (N,K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension

        key:  a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        value: a Tensor of shape (N, K, M) where N is the batch size, K is the
            sequence length and M is the sequence embeding dimension


        mask: a Bool Tensor of shape (N, K, K) that is used to mask the weights
            used for computing weighted sum of values


    return:
        y: a tensor of shape (N, K, M) that contains the weighted sum of values

        weights_softmax: a tensor of shape (N, K, K) that contains the softmaxed
            weight matrix.

    """

    _, _, M = query.shape
    y = None
    weights_softmax = None
    ###############################################################################
    # TODO: This function performs same function as self_attention_two_loop_batch #
    # Implement this function using no loops.                                     #
    # For the mask part, you can ignore it for now and revisit it in the later part.
    # Given the shape of the mask is (N, K, K), and it is boolean with True values#
    # indicating  the weights that have to be masked and False values indicating  #
    # the weghts that dont need to be masked at that position. These masked-scaled#
    # weights can then be softmaxed to compute the final weighted sum of values   #
    # Hint: look at torch.bmm and torch.masked_fill                               #
    ###############################################################################
    # Replace "pass" statement with your code
    
    scores = torch.bmm(query, key.transpose(1, 2)) / torch.sqrt(torch.tensor(M, dtype=torch.float32))
    
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    
    weights_softmax = F.softmax(scores, dim=-1)
    
    y = torch.bmm(weights_softmax, value)
    
    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return y, weights_softmax
    

class SelfAttention(nn.Module):
    def __init__(self, dim_in: int, dim_q: int, dim_v: int):
        super().__init__()

        """
        This class encapsulates the implementation of self-attention layer. We map 
        the input query, key, and value using MLP layers and then use 
        scaled_dot_product_no_loop_batch to the final output.
        
        args:
            dim_in: an int value for input sequence embedding dimension
            dim_q: an int value for output dimension of query and key vector
            dim_v: an int value for output dimension for value vectors

        """
        self.q = nn.Linear(dim_in, dim_q)  # Linear layer for query
        self.k = nn.Linear(dim_in, dim_q)  # Linear layer for key
        self.v = nn.Linear(dim_in, dim_v)  # Linear layer for value

        ##########################################################################
        # TODO: This function initializes three functions to transform the 3 input
        # sequences to key, query and value vectors. More precisely, initialize  #
        # three nn.Linear layers that can transform the input with dimension     #
        # dim_in to query with dimension dim_q, key with dimension dim_q, and    #
        # values with dim_v. For each Linear layer, use the following strategy to#
        # initialize the weights:                                                #
        # If a Linear layer has input dimension D_in and output dimension D_out  #
        # then initialize the weights sampled from a uniform distribution bounded#
        # by [-c, c]                                                             #
        # where c = sqrt(6/(D_in + D_out))                                       #
        # Please use the same names for query, key and value transformations     #
        # as given above. self.q, self.k, and self.v respectively.               #
        ##########################################################################
        
        # Initialize weights and biases for k, q, v
        init_f = torch.sqrt(torch.tensor(6.0 / (dim_in + dim_q))) #init factor
        
        nn.init.uniform_(self.q.weight, - init_f, init_f)
        nn.init.zeros_(self.q.bias)
        
        nn.init.uniform_(self.k.weight, - init_f, init_f)
        nn.init.zeros_(self.k.bias)
        
        init_v = torch.sqrt(torch.tensor(6.0 / (dim_in + dim_v)))
        nn.init.uniform_(self.v.weight, - init_v, init_v)
        nn.init.zeros_(self.v.bias)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the self-attention layer.

        args:
            query: Tensor of shape (N, K, M)
            key: Tensor of shape (N, K, M)
            value: Tensor of shape (N, K, M)
            mask: Tensor of shape (N, K, K)
        return:
            y: Tensor of shape (N, K, dim_v)
        """
        self.weights_softmax = (
            None  # weight matrix after applying self_attention_no_loop_batch
        )
        y = None
        ##########################################################################
        # TODO: Use the functions initialized in the init fucntion to find the   #
        # output tensors. Precisely, pass the inputs query, key and value to the #
        #  three functions iniitalized above. Then, pass these three transformed #
        # query,  key and value tensors to the self_attention_no_loop_batch to   #
        # get the final output. For now, dont worry about the mask and just      #
        # pass it as a variable in self_attention_no_loop_batch. Assign the value#
        # of output weight matrix from self_attention_no_loop_batch to the       #
        # variable self.weights_softmax                                          #
        ##########################################################################
        
        # Linear transformation of query, key, and value
        query_transformed = self.q(query)
        key_transformed = self.k(key)
        value_transformed = self.v(value)
        
        y, self.weights_softmax = scaled_dot_product_no_loop_batch(query_transformed, 
        key_transformed, value_transformed, mask)
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return y


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_out: int):
        super().__init__()

        """
        
        A naive implementation of the MultiheadAttention layer for Transformer model.
        We use multiple SelfAttention layers parallely on the same input and then concat
        them to into a single tensor. This Tensor is then passed through an MLP to 
        generate the final output. The input shape will look like (N, K, M) where  
        N is the batch size, K is the batch size and M is the sequence embedding  
        dimension.
        args:
            num_heads: int value specifying the number of heads
            dim_in: int value specifying the input dimension of the query, key
                and value. This will be the input dimension to each of the
                SingleHeadAttention blocks
            dim_out: int value specifying the output dimension of the complete 
                MultiHeadAttention block



        NOTE: Here, when we say dimension, we mean the dimesnion of the embeddings.
              In Transformers the input is a tensor of shape (N, K, M), here N is
              the batch size , K is the sequence length and M is the size of the
              input embeddings. As the sequence length(K) and number of batches(N)
              don't change usually, we mostly transform
              the dimension(M) dimension.


        """

        ##########################################################################
        # TODO: Initialize two things here:                                      #
        # 1.) Use nn.ModuleList to initialze a list of SingleHeadAttention layer #
        # modules.The length of this list should be equal to num_heads with each #
        # SingleHeadAttention layer having input dimension as dim_in, and query  #
        # , key, and value dimension as dim_out.                                 #
        # 2.) Use nn.Linear to map the output of nn.Modulelist block back to     #
        # dim_in. Initialize the weights using the strategy mentioned in         #
        # SelfAttention.                                                         #
        ##########################################################################
        
        self.heads = nn.ModuleList([
            SelfAttention(dim_in, dim_out, dim_out) for _ in range(num_heads)
        ])
        self.out = nn.Linear(num_heads * dim_out, dim_in)
        
        # Initialize weights & biases
        init_f = torch.sqrt(torch.tensor((6.0 / (num_heads*dim_out + dim_in))))        
        torch.nn.init.uniform_(self.out.weight, a=-init_f, b=init_f)
        torch.nn.init.constant_(self.out.bias, 0)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        An implementation of the forward pass of the MultiHeadAttention layer.

        args:
            query: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            key: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            value: Tensor of shape (N, K, M) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

            mask: Tensor of shape (N, K, K) where N is the number of sequences in
                the batch, K is the sequence length and M is the input embedding
                dimension. M should be equal to dim_in in the init function

        returns:
            y: Tensor of shape (N, K, M)
        """
        y = None
        ##########################################################################
        # TODO: You need to perform a forward pass through the MultiHeadAttention#
        # block using the variables defined in the initializing function. The    #
        # nn.ModuleList behaves as a list and you could use a for loop or list   #
        # comprehension to extract different elements of it. Each of the elements#
        # inside nn.ModuleList is a SingleHeadAttention that  will take the same #
        # query, key, value and mask tensors and you will get a list of tensors as
        # output. Concatenate this list if tensors and pass them through the     #
        # nn.Linear mapping function defined in the initialization step.         #
        ##########################################################################
        
        head_outputs = [head(query, key, value, mask) for head in self.heads] # SingleAtt outputs
        concatenated = torch.cat(head_outputs, dim=2)
        y = self.out(concatenated)
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y



class LayerNormalization(nn.Module):
    def __init__(self, emb_dim: int, epsilon: float = 1e-10):
        super().__init__()

        """
        The class implements the Layer Normalization for Linear layers in 
        Transformers.  Unlike BatchNorm, it estimates the normalization statistics 
        for each element present in the batch and hence does not depend on the  
        complete batch.
        The input shape will look something like (N, K, M) where N is the batch 
        size, K is the sequence length, and M is the sequence length embedding. 
        We compute the mean with shape (N, K) and standard deviation with shape 
        (N, K) and use them to normalize each sequence.
        
        args:
            emb_dim: int representing embedding dimension
            epsilon: float value

        """

        self.epsilon = epsilon

        ##########################################################################
        # TODO: Initialize the scale and shift parameters for LayerNorm.         #
        # Initialize the scale parameters to all ones and shift parameter to all #
        # zeros. As we have seen in the lecture, the shape of scale and shift    #
        # parameters remains the same as in Batchnorm, initialize these parameters
        # with appropriate dimensions. Don't forget to encapsulate these scale and#
        # shift initializations with nn.Parameter                                #
        ##########################################################################

        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.beta = nn.Parameter(torch.zeros(emb_dim))

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x: Tensor):
        """
        An implementation of the forward pass of the Layer Normalization.

        args:
            x: a Tensor of shape (N, K, M) or (N, K) where N is the batch size, K
                is the sequence length, and M is the embedding dimension

        returns:
            y: a Tensor of shape (N, K, M) or (N, K) after applying layer
                normalization

        """
        y = None
        
        ##########################################################################
        # TODO: Implement the forward pass of the LayerNormalization layer.      #
        # Compute the mean and standard deviation of input and use these to      #
        # normalize the input. Further, use self.gamma and self.beta to scale    #
        # and shift this normalized input.                                      #
        ##########################################################################

        mean = x.mean(dim=-1, keepdim=True)
        std = (x - mean).pow(2).mean(dim=-1, keepdim=True).add(self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.gamma + self.beta

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class FeedForwardBlock(nn.Module):
    def __init__(self, inp_dim: int, hidden_dim_feedforward: int):
        super().__init__()

        """
        An implementation of the FeedForward block in the Transformers. We pass  
        the input through stacked 2 MLPs and 1 ReLU layer. The forward pass has  
        following architecture:
        
        linear - relu -linear
        
        The input will have a shape of (N, K, M) where N is the batch size, K is 
        the sequence length and M is the embedding dimension. 
        
        args:
            inp_dim: int representing embedding dimension of the input tensor
                     
            hidden_dim_feedforward: int representing the hidden dimension for
                the feedforward block
        """

        ##########################################################################
        # TODO: initialize two MLPs here with the first one using inp_dim as input
        # dimension and hidden_dim_feedforward as output and the second with     #
        # hidden_dim_feedforward as input. You should figure out the output      #
        # dimension of the second MLP. Initialize the weights of all the MLPs    #
        # according to the strategy mentioned in SelfAttention block.            #
        ##########################################################################
        
        self.linear1 = nn.Linear(inp_dim, hidden_dim_feedforward)
        self.linear2 = nn.Linear(hidden_dim_feedforward, inp_dim)
        self.relu = nn.ReLU()
        
        # Initialize weights & biases
        init_f = torch.sqrt(torch.tensor((6.0 / (inp_dim + hidden_dim_feedforward))))
        torch.nn.init.uniform_(self.linear1.weight, a=-init_f, b=init_f)
        torch.nn.init.constant_(self.linear1.bias, 0)

        init_f_k = init_f
        torch.nn.init.uniform_(self.linear2.weight, a=-init_f_k, b=init_f_k)
        torch.nn.init.constant_(self.linear2.bias, 0)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x):
        """
        An implementation of the forward pass of the FeedForward block.

        args:
            x: a Tensor of shape (N, K, M) which is the output of
               MultiHeadAttention
        returns:
            y: a Tensor of shape (N, K, M)
        """
        y = None
        ###########################################################################
        # TODO: Use the two MLP layers initialized in the init function to perform#
        # a forward pass. You should be using a ReLU layer after the first MLP and#
        # no activation after the second MLP.                                     #
        ###########################################################################
        
        y = self.linear1(x)
        y = self.relu(y)
        y = self.linear2(y)
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y



class EncoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        """
        This class implements the encoder block for the Transformer model, the 
        original paper used 6 of these blocks sequentially to train the final model. 
        Here, we will first initialize the required layers using the building  
        blocks we have already  implemented, and then finally write the forward     
        pass using these initialized layers, residual connections and dropouts.        
        
        As shown in the Figure 1 of the paper attention is all you need
        https://arxiv.org/pdf/1706.03762.pdf, the encoder consists of four components:
        
        1. MultiHead Attention
        2. FeedForward layer
        3. Residual connections after MultiHead Attention and feedforward layer
        4. LayerNorm
        
        The architecture is as follows:
        
       inp - multi_head_attention - out1 - layer_norm(out1 + inp) - dropout - out2 \ 
        - feedforward - out3 - layer_norm(out3 + out2) - dropout - out
        
        Here, inp is input of the MultiHead Attention of shape (N, K, M), out1, 
        out2 and out3 are the outputs of the corresponding layers and we add these 
        outputs to their respective inputs for implementing residual connections.

        args:
            num_heads: int value specifying the number of heads in the
                MultiHeadAttention block of the encoder

            emb_dim: int value specifying the embedding dimension of the input
                sequence

            feedforward_dim: int value specifying the number of hidden units in the 
                FeedForward layer of Transformer

            dropout: float value specifying the dropout value


        """

        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )
        ##########################################################################
        # TODO: Initialize the following layers:                                 #
        # 1. One MultiHead Attention block using num_heads as number of heads and#
        #    emb_dim as the input dimension. You should also be able to compute  #
        #    the output dimension of MultiheadHead attention given num_heads and #
        #    emb_dim.                                                            #
        #    Hint: use the logic that you concatenate the output from each       #
        #    SingleHeadAttention inside the MultiHead Attention block and choose #
        #    the output dimension such that the concatenated tensor and the input#
        #    tensor have the same embedding dimension.                           #
        #                                                                        #
        # 2. Two LayerNorm layers with input dimension equal to emb_dim          #
        # 3. One feedForward block taking input as emb_dim and hidden units as   #
        #    feedforward_dim                                                     #
        # 4. A Dropout layer with given dropout parameter                        #
        ##########################################################################
        # Replace "pass" statement with your code
        
        self.multihead_attention = MultiHeadAttention(num_heads, emb_dim, emb_dim//num_heads)
        self.layer_norm1 = LayerNormalization(emb_dim)
        self.feedforward_block = FeedForwardBlock(emb_dim, feedforward_dim)
        self.layer_norm2 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(self, x):
        """
        An implementation of the forward pass of the EncoderBlock of the
        Transformer model.
        args:
            x: a Tensor of shape (N, K, M) as input sequence
        returns:
            y: a Tensor of shape (N, K, M) as the output of the forward pass
        """
        y = None
        ##########################################################################
        # TODO: Use the layer initialized in the init function to complete the   #
        # forward pass. As Multihead Attention takes in 3 inputs, use the same   #
        # input thrice as the input. Follow the Figure 1 in Attention is All you #
        # Need paper to complete the rest of the forward pass. You can also take #
        # reference from the architecture written in the fucntion documentation. #
        ##########################################################################
        # Replace "pass" statement with your code
        
        attention_output = self.multihead_attention(x, x, x)    # Multi-Head Attention out
        x = x + attention_output                                # Residual connection
        x = self.layer_norm1(x) 
        x = self.dropout(x)
        
        feedforward_output = self.feedforward_block(x)          # Feed Forward Block out
        x = x + feedforward_output                              # Residual connection
        x = self.layer_norm2(x)
        y = self.dropout(x)
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return y




def get_subsequent_mask(seq):
    """
    An implementation of the decoder self attention mask. This will be used to
    mask the target sequence while training the model. The input shape here is
    (N, K) where N is the batch size and K is the sequence length.

    args:
        seq: a tensor of shape (N, K) where N is the batch size and K is the
             length of the sequence
    return:
        mask: a tensor of shape (N, K, K) where N is the batch size and K is the
              length of the sequence

    Given a sequence of length K, we want to mask the weights inside the function
    `self_attention_no_loop_batch` so that it prohibits the decoder to look ahead
    in the future
    """
    mask = None
    ###############################################################################
    # TODO: This function constructs mask for the decoder part of the Transformer.#
    # To implement this, for each sequence (of K) in the batch(N) return a        #
    # boolean matrix that is True for the place where we have to apply mask and   #
    # False where we don't have to apply the mask.                                #
    #                                                                             #
    ###############################################################################
    # Replace "pass" statement with your code

    N, K = seq.shape
    
    mask = torch.triu(torch.ones((K, K), dtype=torch.bool, device=seq.device), diagonal=1)
    mask = mask.unsqueeze(0).expand(N, -1, -1)

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    return mask



class DecoderBlock(nn.Module):
    def __init__(
        self, num_heads: int, emb_dim: int, feedforward_dim: int, dropout: float
    ):
        super().__init__()
        if emb_dim % num_heads != 0:
            raise ValueError(
                f"""The value emb_dim = {emb_dim} is not divisible
                             by num_heads = {num_heads}. Please select an
                             appropriate value."""
            )

        ##########################################################################
        # TODO: Initialize the following layers:                                 #
        # 1. Two MultiheadAttention layers with num_heads number of heads, emb_dim
        #     as the embedding dimension. As done in Encoder, you should be able to
        #     figure out the output dimension of both the MultiHeadAttention.    #
        # 2. One FeedForward block that takes in emb_dim as input dimension and  #
        #   feedforward_dim as hidden layers                                     #
        # 3. LayerNormalization layers after each of the block                   #
        # 4. Dropout after each of the block                                     #
        ##########################################################################

        self.attention_self = MultiHeadAttention(num_heads, emb_dim, emb_dim//num_heads)
        self.attention_cross = MultiHeadAttention(num_heads, emb_dim, emb_dim//num_heads)
        self.feedforward = FeedForwardBlock(emb_dim, feedforward_dim)
        self.norm1 = LayerNormalization(emb_dim)
        self.norm2 = LayerNormalization(emb_dim)        
        self.norm3 = LayerNormalization(emb_dim)
        self.dropout = nn.Dropout(dropout)

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

    def forward(
        self, dec_inp: Tensor, enc_inp: Tensor, mask: Tensor = None
    ) -> Tensor:

        """
        args:
            dec_inp: a Tensor of shape (N, K, M)
            enc_inp: a Tensor of shape (N, K, M)
            mask: a Tensor of shape (N, K, K)

        This function will handle the forward pass of the Decoder block. It takes
        in input as enc_inp which is the encoder output and a tensor dec_inp which
        is the target sequence shifted by one in case of training and an initial
        token "BOS" during inference
        """
        y = None

        ##########################################################################
        # TODO: Using the layers initialized in the init function, implement the #
        # forward pass of the decoder block. Pass the dec_inp to the             #
        # self.attention_self layer. This layer is responsible for the self      #
        # interation of the decoder input. You should follow the Figure 1 in     #
        # Attention is All you need paper to implenment the rest of the forward  #
        # pass. Don't forget to apply the residual connections for different layers.
        ##########################################################################

        # Masked Self Attention Block
        out1 = self.attention_self(dec_inp, dec_inp, dec_inp, mask)
        out2 = out1 + dec_inp           # Residual Connection
        out3 = self.norm1(out2)
        out4 = self.dropout(out3)              

        # Cross Attention Block
        out5 = self.attention_cross(out4, enc_inp, enc_inp)
        out6 = out4 + out5              # Residual Connection
        out7 = self.norm2(out6)
        out8 = self.dropout(out7)

        # Feed Forward Block
        out9 = self.feedforward(out8)
        out10 = out8 + out9             # Residual Connection
        out11 = self.norm3(out10)
        out12 = self.dropout(out11)
        
        y = out12

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        return y


class Encoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
    ):
        """
        The class encapsulates the implementation of the final Encoder that use
        multiple EncoderBlock layers.

        args:
            num_heads: int representing number of heads to be used in the
                EncoderBlock
            emb_dim: int repreesenting embedding dimension for the Transformer
                model
            feedforward_dim: int representing hidden layer dimension for the
                feed forward block

        """

        super().__init__()
        self.layers = nn.ModuleList(
            [
                EncoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src_seq: Tensor):
        for _layer in self.layers:
            src_seq = _layer(src_seq)

        return src_seq


class Decoder(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        num_layers: int,
        dropout: float,
        vocab_len: int,
    ):
        super().__init__()
        """
        The Decoder takes the input from the encoder and the target
        sequence to generate the final sequence for the output. We
        first pass the input through stacked DecoderBlocks and then
        project the output to vocab_len which is required to get the
        actual sequence.
        
        args:
            num_heads: Int representing number of heads in the MultiheadAttention
            for Transformer
            emb_dim: int representing the embedding dimension
            of the sequence
            feedforward_dim: hidden layers in the feed forward block
            num_layers: int representing the number of DecoderBlock in Decoder
            dropout: float representing the dropout in each DecoderBlock
            vocab_len: length of the vocabulary


        """

        self.layers = nn.ModuleList(
            [
                DecoderBlock(num_heads, emb_dim, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.proj_to_vocab = nn.Linear(emb_dim, vocab_len)
        a = (6 / (emb_dim + vocab_len)) ** 0.5
        nn.init.uniform_(self.proj_to_vocab.weight, -a, a)

    def forward(self, target_seq: Tensor, enc_out: Tensor, mask: Tensor):

        out = target_seq.clone()
        for _layer in self.layers:
            out = _layer(out, enc_out, mask)
        out = self.proj_to_vocab(out)
        return out



def position_encoding_simple(K: int, M: int) -> Tensor:
    """
    An implementation of the simple positional encoding using uniform intervals
    for a sequence.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)
    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K, construct a 1D Tensor of length#
    # K with nth element as n/K, where n starts from 0. Replicate this tensor M  #
    # times to create a tensor of the required output shape                      #
    ##############################################################################
    # Replace "pass" statement with your code
    
    y = torch.zeros(1, K, M)
    positions = torch.arange(K).unsqueeze(1).float()  # Generate positions from 0 to K-1
    y[0, :, :] = positions / K                        # Divide each position by K to get uniform intervals

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    
    return y



def position_encoding_sinusoid(K: int, M: int) -> Tensor:
    """
    An implementation of the sinusoidal positional encodings.

    args:
        K: int representing sequence length
        M: int representing embedding dimension for the sequence

    return:
        y: a Tensor of shape (1, K, M)
    """
    y = None
    ##############################################################################
    # TODO: Given the length of input sequence K and embedding dimension M       #
    # construct a tesnor of shape (K, M) where the value along the dimensions    #
    # follow the equations given in the notebook. Make sure to keep in mind the  #
    # alternating sines and cosines along the embedding dimension M.             #
    ##############################################################################
    # Replace "pass" statement with your code
    
    y_ = torch.zeros(K,M)
    for p in range(K):
        for i in range(ceil(M/2)):

            a =  floor(2*i/M)         
            y_[p, 2*i]  =  sin(p/(10000)**a) 
            if 2*i+1 < M:
                y_[p, 2*i+1]= cos(p/(10000)**a)

    y = y_.reshape(1,K,M)

    ##############################################################################
    #               END OF YOUR CODE                                             #
    ##############################################################################
    
    return y



class Transformer(nn.Module):
    def __init__(
        self,
        num_heads: int,
        emb_dim: int,
        feedforward_dim: int,
        dropout: float,
        num_enc_layers: int,
        num_dec_layers: int,
        vocab_len: int,
    ):
        super().__init__()

        """
        The class implements Transformer model with encoder and decoder. The input
        to the model is a tensor of shape (N, K) and the output is a tensor of shape
        (N*O, V). Here, N is the batch size, K is the input sequence length, O is  
        the output sequence length and V is the Vocabulary size. The input is passed  
        through shared nn.Embedding layer and then added to input positonal 
        encodings. Similarily, the target is passed through the same nn.Embedding
        layer and added to the target positional encodings. The only difference
        is that we take last but one  value in the target. The summed 
        inputs(look at the code for detials) are then sent through the encoder and  
        decoder blocks  to get the  final output.
        args:
            num_heads: int representing number of heads to be used in Encoder
                       and decoder
            emb_dim: int representing embedding dimension of the Transformer
            dim_feedforward: int representing number of hidden layers in the
                             Encoder and decoder
            dropout: a float representing probability for dropout layer
            num_enc_layers: int representing number of encoder blocks
            num_dec_layers: int representing number of decoder blocks

        """
        self.emb_layer = None
        ##########################################################################
        # TODO: Initialize an Embedding layer mapping vocab_len to emb_dim. This #
        # is the very first input to our model and transform this input to       #
        # emb_dim that will stay the same throughout our model. Please use the   #
        # name of this layer as self.emb_layer                                   #
        ##########################################################################
        
        self.emb_layer = nn.Embedding(vocab_len, emb_dim)  # Initialize the embedding layer

        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################
        self.encoder = Encoder(
            num_heads, emb_dim, feedforward_dim, num_enc_layers, dropout
        )
        self.decoder = Decoder(
            num_heads,
            emb_dim,
            feedforward_dim,
            num_dec_layers,
            dropout,
            vocab_len,
        )

    def forward(
        self, ques_b: Tensor, ques_pos: Tensor, ans_b: Tensor, ans_pos: Tensor
    ) -> Tensor:

        """

        An implementation of the forward pass of the Transformer.

        args:
            ques_b: Tensor of shape (N, K) that consists of input sequence of
                the arithmetic expression
            ques_pos: Tensor of shape (N, K, M) that consists of positional
                encodings of the input sequence
            ans_b: Tensor of shape (N, K) that consists of target sequence
                of arithmetic expression
            ans_pos: Tensor of shape (N, K, M) that consists of positonal
                encodings of the target sequence

        returns:
            dec_out: Tensor of shape (N*O, M) where O is the size of
                the target sequence.
        """
        q_emb = self.emb_layer(ques_b)
        a_emb = self.emb_layer(ans_b)
        q_emb_inp = q_emb + ques_pos
        a_emb_inp = a_emb[:, :-1] + ans_pos[:, :-1]
        dec_out = None
        ##########################################################################
        # TODO: This portion consists of writing the forward part for the complete
        # Transformer. First, pass the q_emb_inp through the encoder, this will be
        # the encoder output which you should use as one of the decoder inputs.
        # Along with the encoder output, you should also construct an appropriate
        # mask using the get_subsequent_mask. Finally, pass the a_emb_inp, the
        # encoder output and the mask to the decoder. The task here is to mask
        # the values of the target(a_emb_inp)
        # Hint: the mask shape will depend on the Tensor ans_b
        ##########################################################################
        
        enc_out = self.encoder(q_emb_inp)

        mask = get_subsequent_mask(ans_b[:,:-1])

        dec_out = self.decoder(a_emb_inp, enc_out, mask)
        dec_out = dec_out.reshape(dec_out.shape[0]*dec_out.shape[1],-1 )
        
        ##########################################################################
        #               END OF YOUR CODE                                         #
        ##########################################################################

        return dec_out


class AddSubDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        input_seqs,
        target_seqs,
        convert_str_to_tokens,
        special_tokens,
        emb_dim,
        pos_encode,
    ):

        """
        The class implements the dataloader that will be used for the toy dataset.

        args:
            input_seqs: A list of input strings
            target_seqs: A list of output strings
            convert_str_to_tokens: Dictionary to convert input string to tokens
            special_tokens: A list of strings
            emb_dim: embedding dimension of the transformer
            pos_encode: A function to compute positional encoding for the data
        """

        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.convert_str_to_tokens = convert_str_to_tokens
        self.emb_dim = emb_dim
        self.special_tokens = special_tokens
        self.pos_encode = pos_encode

    def preprocess(self, inp):
        return prepocess_input_sequence(
            inp, self.convert_str_to_tokens, self.special_tokens
        )

    def __getitem__(self, idx):
        """
        The core fucntion to get element with index idx in the data.
        args:
            idx: index of the element that we need to extract from the data
        returns:
            preprocess_inp: A 1D tensor of length K, where K is the input sequence
                length
            inp_pos_enc: A tensor of shape (K, M), where K is the sequence length
                and M is the embedding dimension
            preprocess_out: A 1D tensor of length O, where O is the output
                sequence length
            out_pos_enc: A tensor of shape (O, M), where O is the sequence length
                and M is the embedding dimension
        """

        inp = self.input_seqs[idx]
        out = self.target_seqs[idx]
        preprocess_inp = torch.tensor(self.preprocess(inp))
        preprocess_out = torch.tensor(self.preprocess(out))
        inp_pos = len(preprocess_inp)
        inp_pos_enc = self.pos_encode(inp_pos, self.emb_dim)
        out_pos = len(preprocess_out)
        out_pos_enc = self.pos_encode(out_pos, self.emb_dim)

        return preprocess_inp, inp_pos_enc[0], preprocess_out, out_pos_enc[0]

    def __len__(self):
        return len(self.input_seqs)


def LabelSmoothingLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    ground = ground.contiguous().view(-1)
    eps = 0.1
    n_class = pred.size(1)
    one_hot = torch.nn.functional.one_hot(ground).to(pred.dtype)
    one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
    log_prb = F.log_softmax(pred, dim=1)
    loss = -(one_hot * log_prb).sum(dim=1)
    loss = loss.sum()
    return loss


def CrossEntropyLoss(pred, ground):
    """
    args:
        pred: predicted tensor of shape (N*O, V) where N is the batch size, O
            is the target sequence length and V is the size of the vocab
        ground: ground truth tensor of shape (N, O) where N is the batch size, O
            is the target sequence
    """
    loss = F.cross_entropy(pred, ground, reduction="sum")
    return loss
