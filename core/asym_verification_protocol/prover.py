__author__ = 'gru'

# General libs
from collections import namedtuple
import string
import random

import numpy as np
import scipy.sparse as sps


# Crypto lib
from Crypto import Random
from Crypto.Cipher import AES
from Crypto.Hash import SHA512

# Watermarking lib
from core.audio_cwe.xs_wm_scheme import XsWMSystem
from core.asym_verification_protocol import permutation_utils as utils


OwnershipTicket = namedtuple("OwnershipTicket", "c1, c2, hash_g_i")


class ProtocolError(Exception):
    """
    Indicates an error in the course of the verification protocol.
    For example a ProtocolError is raised, when the verifier asks the prover
    to reveal both commitments, etc...
    """
    pass


class Prover:
    """
    This class extends a histogram-based watermarking scheme (e.g.
    Schmitz', Xiang's or the combined method) to enable minimum knowledge
    verification of the mark.

    G and \tau(G) act as public key and the
    permutation \tau as the private key. In an iterative process \tau is
    separated in two partial permutations \rho and \sigma. These
    permutations are used to construct a different scramblings of the mark
    wmk_i and the graph G_i. With the help of cryptographic commitments (c1
    and c2 based on symmetric encryption) one of these permutations is
    revealed to the verifier, who can then check the validity of either
    the presence of the mark_i in the object or the prover's knowledge of \tau.
    """

    def __init__(self, tau_iv, size):
        """Constructs a Prover-object.

        :param tau_iv: the logistic map parameters, that are used to
        generate tau
        :param size: the size of the graph and therefore also the size of
        the watermarks, that can be verified with the prover's public key
        :return:None
        """
        self._tau = None

        self._graph = None
        self._graph_size = None
        self._graph_scrambled = None

        self._obj = None
        self._obj_marked = None

        self._mark = None
        self._mark_scrambled = None

        self._seed = None
        self._detection_key = None
        self._wm_key_scrambled = None
        self._wm_params = None

        self._c1_params = None
        self._c2_params = None

        self._can_reveal = False
        self._is_init = False

        self._generate_public_key(tau_iv, size)

    def prepare(self, obj, iv, mark, key):
        """Prepare the prover for a verification process by setting the
        object, the initialization vector of the watermarking system,
        the mark and the detection key.

        :param obj: a list of samples representing the media object
        :param iv: a dict representing the initialization vector
        :param mark: the watermark
        :param key: a list of bin pairs representing the detection key
        :return:None
        """
        # store params
        self._mark = mark
        self._obj_marked = obj
        self._detection_key = key
        self._wm_params = iv

        # create scrambled watermark and scrambled watermark key
        self._wm_key_scrambled = utils.permute_list(self._detection_key,
                                                    self._tau)
        self._mark_scrambled = utils.permute_list(self._mark, self._tau)
        self._is_init = True


    def emb(self, o, m, embedding_key, **kwargs):
        """Implements the emb()-method of the proposed protocol.
        Embeds a watermark m into an media-object - in this case an audio
        file -  via the combined method (Xiang+Schmitz) and stores all for
        the verification process necessary information in the Prover-object,
        on which it was called. The seed serves as embedding key.

        :param o: the media object to mark - a list of samples
        :param m: the watermark - a list of bits
        :param embedding_key: a scalar, which serves as seed and therefore
        as the
        embedding key
        :param kwargs: the initialization parameters for the watermarking
        system. See XsWMSystem.__init__(..)
        :return:
        """
        if len(m) != self._graph_size:
            raise ValueError('Length of WMK has to match length of public key')

        # store params
        self._mark = m
        self._obj = o
        self._seed = embedding_key

        # Initialize watermarking system
        wm_sys = XsWMSystem(**kwargs)

        # Mark the cover work
        self._obj_marked, self._detection_key = wm_sys.embed_watermark(
            self._obj, self._mark, key=self._seed)
        self._wm_params = wm_sys.get_params()

        # create scrambled watermark and scrambled watermark key
        self._wm_key_scrambled = utils.permute_list(self._detection_key,
                                                    self._tau)
        self._mark_scrambled = utils.permute_list(self._mark, self._tau)
        self._is_init = True

    def ver(self, nonce):
        """Implements the verification method of the proposed protocol. When
        the verifier calls ver(), one protocol iteration is initiated. The
        partial permutations \rho and \sigma are generated, the scrambled
        version of the graph - g_i - is formed and hashed. The commitments
        c1 and c2 are constructed and the ownership ticket returned to the
        caller.

        :return: ot - the ownership ticket OT=<c1, c2, hash(g_i)>
        """
        # Generate rho and sigma
        rho, sigma = self._generate_partial_permutations()

        # Obtain g_i
        scrambled_graph_i = utils.permute_graph(self._graph, rho)

        # Generate hash
        hash_g_i = SHA512.new(
            scrambled_graph_i.toarray().tostring()).hexdigest()

        # Form OT
        ot = self._form_ot(rho, sigma, hash_g_i, nonce)
        self._can_reveal = True

        return ot

    def reveal(self, choice):
        """Reveals the information necessary to decrpyt one of the
        commitments, which are based on symmetric encryption.

        :param choice: a string, either 'c1' or 'c2'
        :return: _c1_params or _c2_params: the parameters (IV; key, padding
        length) to decrypt commitment c1 respectively c2
        """
        # Make sure, that it can't be called twice
        if self._can_reveal:
            self._can_reveal = False

            if choice == self.CHOICES[0]:
                return self._c1_params

            elif choice == self.CHOICES[1]:
                return self._c2_params
            else:
                raise ProtocolError('Protocol choice not understood')
        else:
            raise ProtocolError('Reveal was illegally called twice in one '
                                'iteration.')

    def _generate_partial_permutations(self):
        """Generates perm_a(\rho) and perm_b (\sigma). At first \rho is
        constructed. Then \sigma is so formed, that \sigma \circ \rho = \tau
        is valid.

        :return: perm_a, perm_b: \rho and \sigma
        """
        # Avoid identity permutation
        perm_a = utils.generate_random_permutation(self._graph_size)

        # Generate sigma
        perm_b = utils.compose_permutations(self._tau, utils.invert(perm_a))

        return perm_a, perm_b

    CHOICES = ['c1', 'c2']

    def _form_ot(self, rho_perm, sigma_perm, hash_g_i, rand):
        """Constructs the commitments by taking the received random
        number and appending it to each permutation. Theses are then
        encrypted and stored in the ownership ticket together with the hash
        of g_i.

        :param rho_perm: \rho
        :param sigma_perm: \sigma
        :param hash_g_i: the hash of \rho(g))g_i
        :param rand: the nonce given by the verifier
        :return: ot: the ownership ticket
        """
        # Append nonce to each permutation and encrypt with AES
        c1, self._c1_params = self._aes_encrypt_permutation(
            np.append(sigma_perm, rand))
        c2, self._c2_params = self._aes_encrypt_permutation(
            np.append(rho_perm, rand))

        return OwnershipTicket(c1, c2, hash_g_i)

    @staticmethod
    def _aes_encrypt_permutation(p):
        """Encrypts a permutation p by the means of the symmetric encryption
        standard AES. To achieve that, the permutation is converted to a
        string, where a ',' separates the list elements. Then padding is
        applied to have a plaintext length, which is a multiple to the
        blocksize of 16 bytes.

        :param p: a ndarray/list -the permutation to encrypt
        :return: cipher_text: the ciphered permutation
                enc_params: the encryption parameters (iv, key, padding_length)
        """
        key_length = 24  # 192 Bit key size
        IV_LENGTH = 16
        BLOCK_SIZE = 16
        key = Random.get_random_bytes(key_length)
        iv = Random.get_random_bytes(IV_LENGTH)
        aes_obj = AES.new(key, AES.MODE_CBC, iv)

        message = p.tostring()
        len_padding = 0

        while len(message) % BLOCK_SIZE != 0:
            message += random.choice(string.ascii_letters).encode()
            len_padding += 1

        enc_params = [key, iv, len_padding]
        cipher_text = aes_obj.encrypt(message)

        return cipher_text, enc_params

    def setup_verification(self):
        """Returns all necessary information to perform the verification to
        the caller.

        :return: the marked object, the public key, the scrambled wmk-key
        and the watermarking parameters
        """
        return self._obj_marked, self._graph, self._graph_scrambled, \
               self._mark_scrambled, self._wm_key_scrambled, self._wm_params

    def _generate_public_key(self, tau_iv, graph_size):
        """Generates G, \tau and \tau(G) and stores them in instance
        variables. G and \tau(G) form the public and \tau the secret key.
        The difficulty to deduce \tau from the public parameters is based
        on the GI-Problem. !The public key should be signed by a CA!

        :param tau_iv: the parameters for the logistic map to generate \tau
        :param graph_size: the number of nodes in the graph
        :return: NOne
        """
        self._tau = utils.generate_permutation(graph_size, *tau_iv)
        identity = utils.identity(graph_size)

        assert not np.array_equal(self._tau, identity), 'Tau == Identity'

        # create directed, weighted graph which is a compressed sparse row
        # matrix, which forms the adjacency matrix
        self._graph = sps.rand(graph_size, graph_size, density=0.2,
                               format='csr')

        self._graph_size = graph_size

        # permute graph
        self._graph_scrambled = utils.permute_graph(self._graph, self._tau)