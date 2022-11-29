import tensorflow as tf
from tensorflow import keras
from keras import layers


class ReviewClassifier(tf.keras.Model):
    def __init__(self, maxlen, vocab_size, embed_dim, num_heads, ff_dims,
                 reg=1e-3, dropout=0.3):
        super().__init__()

        # Token embedding
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,
                                          mask_zero=True, name="t_emb")

        # Position embedding
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim,
                                        name="p_emb")

        # Self-attention based on token and positional info
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim,
                                             name="attention")

        # Feed-forward network with a layer for each number of neurons in ff_dims
        ffn_layers = []
        for dim in ff_dims:
            layer = layers.Dense(dim, activation="relu",
                                 kernel_regularizer=keras.regularizers.l2(reg))
            ffn_layers.append(layer)
        self.ffn = keras.Sequential(ffn_layers)

        # Normalize activation for speed
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # Dropout for regularization
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

        # Average
        self.pool = layers.GlobalAveragePooling1D()
        self.dropout3 = layers.Dropout(dropout)
        self.dense = layers.Dense(20, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(reg))
        self.dropout4 = layers.Dropout(dropout)

        # 5-star review probability output
        self.stars = layers.Dense(5, activation="softmax")

    def call(self, x, training=False, return_att=False):
        # Create boolean mask for attention to ignore padding
        mask = self.token_emb.compute_mask(x)
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        # Embed tokens and positional encodings
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        x = x + positions

        # Self attention (query = key) -> feed-forward network (w/ add & norm)
        attn_output, attn_scores = self.att(x, x, attention_mask=mask, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        x = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(x)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = self.layernorm2(x + ffn_output)

        # Average pooling (along each time step), dropout, one more dense layer
        x = self.pool(x)
        x = self.dropout3(x)
        x = self.dense(x)
        x = self.dropout4(x)

        # Softmax for 5-star rating probabilities
        rating_probs = self.stars(x)

        if return_att:
            return rating_probs, attn_scores

        return rating_probs
