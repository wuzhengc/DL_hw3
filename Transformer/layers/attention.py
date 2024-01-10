import torch
import torch.nn as nn
import numpy as np

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        # Construct a new MultiHeadAttention layer.

        # Inputs:
        #  - embed_dim: Dimension of the token embedding
        #  - num_heads: Number of attention heads
        #  - dropout: Dropout probability
        super().__init__()
        assert embed_dim % num_heads == 0

        # We will initialize these layers for you, since swapping the ordering
        # would affect the random number generation (and therefore your exact
        # outputs relative to the autograder). Note that the layers use a bias
        # term, but this isn't strictly necessary (and varies by implementation).
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.multi_head_combine = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        self.n_head = num_heads
        self.emd_dim = embed_dim
        self.head_dim = self.emd_dim // self.n_head

    def forward(self, q_data, k_data, v_data, attn_mask=None):
        # Calculate a forward pass of the MultiHeadAttention layer.

        # In the shape definitions below, N is the batch size, S is the source
        # sequence length, T is the target sequence length, and E is the embedding
        # dimension.

        # Inputs:
        # - q_data: Input data to be used as the query, of shape (N, S, E)
        # - k_data: Input data to be used as the key, of shape (N, T, E)
        # - v_data: Input data to be used as the value, of shape (N, T, E)
        # - attn_mask: Array of shape (S, T) where mask[i, j] == 0 indicates token
        #   i in the source should not influence token j in the target.

        # Compute query, key, value *projections* and split into multiple heads.
        # query should be reshaped into (N, H, S, E/H)
        # key and value should be reshaped into (N, H, T, E/H)
        ############################################################################
        # YOUR CODE HERE
        query, key, value = None, None, None

        # query, key, value = None, None, None
        ############################################################################

        output, attention = multi_head_attention(query, key, value, attn_mask=attn_mask, dropout=self.dropout)
        output = self.multi_head_combine(output)

        return output, attention


def multi_head_attention(query, key, value, attn_mask=None, dropout=0.1):
    # Calculate the masked attention output for the provided data, computing
    # all attention heads in parallel.

    # In the shape definitions below, N is the batch size, S is the source
    # sequence length, T is the target sequence length, and E is the embedding
    # dimension.

    # Inputs:
    # - query: Input data to be used as the query, of shape (N, H, S, E/H)
    # - key: Input data to be used as the key, of shape (N, H, T, E/H)
    # - value: Input data to be used as the value, of shape (N, H, T, E/H)
    # - attn_mask: Array of shape (S, T) where mask[i, j] == 0 indicates token
    #   i in the source should not influence token j in the target.

    # Returns:
    # - output: Tensor of shape (N, S, E) giving the weighted combination of
    #   data in value according to the attention weights calculated using key
    #   and query.
    # - attention: Tensor of shape (N, H, T, T) for attention weight matrix for
    #   H heads and T length calculated using key and query.
    N = query.shape[0]
    S = query.shape[2]
    T = key.shape[2]
    E = query.shape[3] * query.shape[1]
    output = torch.empty((N, S, E))
    attention = None

    # Implement multiheaded attention using the equations given in             #
    # homework document.                                                       #
    # A few hints:                                                             #
    #  1) The function torch.matmul allows you to do a batched matrix multiply.#
    #     For example, you can do (N, H, T, E/H) by (N, H, E/H, T) to yield a  #
    #     shape (N, H, T, T). For more examples, see                           #
    #     https://pytorch.org/docs/stable/generated/torch.matmul.html          #
    #  2) For applying attn_mask, think how the scores should be modified to   #
    #     prevent a value from influencing output. Specifically, the PyTorch   #
    #     function masked_fill may come in handy.                              #
    #  3) You can use F.softmax and F.dropout to apply the softmax and dropout #
    #     functions to an entire tensor at once. The default dropout rate of   #
    #     p = 0.1 is fine for our purposes.                                    #
    ############################################################################
    # YOUR CODE HERE
    ############################################################################
    return output, attention

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


if __name__ == '__main__':
    torch.manual_seed(1234)

    # Choose dimensions such that they are all unique for easier debugging:
    # Specifically, the following values correspond to N=1, H=2, T=3, E//H=4, and E=8.
    batch_size = 1
    sequence_length = 3
    embed_dim = 8
    attn = MultiHeadAttention(embed_dim, num_heads=2)

    # Self-attention.
    data = torch.randn(batch_size, sequence_length, embed_dim)
    self_attn_output, output_self_attention = attn(q_data=data, k_data=data, v_data=data)

    # Masked self-attention.
    mask = torch.randn(sequence_length, sequence_length) < 0.5
    masked_self_attn_output, output_masked_self_attention = attn(q_data=data, k_data=data, v_data=data, attn_mask=mask)

    # Attention using two inputs.
    other_data = torch.randn(batch_size, sequence_length, embed_dim)
    attn_output, output_attention = attn(q_data=data, k_data=other_data, v_data=other_data)

    expected_self_attn_output = np.asarray([[
      [-0.30784,-0.08971,0.57260,0.19825,0.08959,0.28221,-0.05153,-0.23268]
      ,[-0.35230,0.10586,0.42247,0.09276,0.13765,0.11636,-0.09490,0.01749]
      ,[-0.30555,-0.23511,0.78333,0.37899,0.26324,0.13141,-0.00239,-0.20351]]])

    expected_masked_self_attn_output = np.asarray([[
      [-0.34701,0.07956,0.40117,-0.00986,0.07048,0.26159,-0.13170,-0.06824]
      ,[-0.26902,-0.53240,0.73553,0.24340,0.12136,0.56356,0.01649,-0.51939]
      ,[-0.23963,-0.00882,0.75761,0.27159,0.16656,0.10638,-0.09657,-0.11547]]])

    expected_attn_output = np.asarray([[
      [-0.483236,0.206833,0.392467,0.031948,0.155175,0.179157,-0.118605,0.049207]
      ,[-0.214869,0.205259,0.261078,0.154042,-0.045083,0.147627,0.077088,-0.050551]
      ,[-0.393120,0.158911,0.252667,0.132215,0.083187,0.254064,0.000776,-0.117547]]])

    print('self_attn_output error: ', rel_error(expected_self_attn_output, self_attn_output.detach().numpy()))
    print('masked_self_attn_output error: ', rel_error(expected_masked_self_attn_output, masked_self_attn_output.detach().numpy()))
    print('attn_output error: ', rel_error(expected_attn_output, attn_output.detach().numpy()))
