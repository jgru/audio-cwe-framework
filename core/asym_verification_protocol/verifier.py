__author__ = 'gru'

import numpy as np
from numpy.random import RandomState
from Crypto.Cipher import AES
from Crypto.Hash import SHA512

from core.asym_verification_protocol.prover import Prover
from core.asym_verification_protocol.prover import ProtocolError
from core.asym_verification_protocol import permutation_utils as utils
from core.audio_cwe.xs_wm_scheme import XsWMSystem


class Verifier():
    """
    This class implements the verifier's part of the proposed
    asymmetric verification scheme.

    In each iteration the verifier
    flips a coin to decide either commitment c1 oder c2. In case (i) he
    can check the presence of the watermark and in case (ii) he can
    check the prover's knowledge of her secret permutation \tau. After i
    successful iterations the verifier can be 100 -(1/(2^i) * 100)% certain,
    that the prover's mark is present in the evaluated audio file.
    """

    def __init__(self, prover, num_rounds=10, seed=0):
        """Intializes a verifier object by setting the counterpart - the
        prover -, specifying the number of iterations and giving the seed
        for the PRNG, which realizes the coin flipping.

        :param prover: an object of typ Prover - the counterpart in the
        protocol
        :param num_rounds: scalar, which specifies the number of
        iterative rounds
        :param seed: scalar, which functions as seed for the PRNG
        :return:
        """
        self.prover = prover
        self._rounds = num_rounds
        self._prng = RandomState(seed)
        self.success = False

    def _flip_coin(self):
        """Chooses an integer value pseudo-randomly between 0 and 1 (
        inclusive)."""
        return self._prng.randint(2)

    def start_verification(self):
        """Performs the actual verification by calling ver() of the prover
        iteratively. In each round the OwnershipTicket, which contains the
        encrypted permutations \rho & \simga and the hash of g_i is received.
        A coin flip decides which commitment should be opened. In case(i)
        the presence of the scrambled watermark is verified and in case(ii)
        the prover's knowledge of \tau is validated. If the verifiers is
        cheaten once, he quits the protocol run.

        :return: success: a boolean which specifies, whether the
        verification process was successfull.
        """
        print('------------------------------------------------------')
        print('Verification process starts')
        print('------------------------------------------------------')

        # Receive public data
        o, g, scrambled_g, scrambled_mark, scrambled_wmk_key, wm_params = \
            self.prover.setup_verification()

        # WM parameters
        print('Watermarking parameters:\n', wm_params)

        has_cheated = False
        i = self._rounds

        while i > 0 and not has_cheated:
            # Get OwnershipTicket
            print('------------------------------------------------------')
            print('Iteration #', self._rounds - i)
            print('------------------------------------------------------')
            nonce = self._prng.randint(np.iinfo('i').max)
            ot = self.prover.ver(nonce)

            if self._flip_coin() is 1:
                print('------------------------------------------------------')
                print("Case (i)")
                print('------------------------------------------------------')
                c1_params = self.prover.reveal(Prover.CHOICES[0])
                sigma_permutation = self._aes_decrypt_array(ot.c1, *c1_params)

                # Check random number at the end of commitment
                if nonce != sigma_permutation[-1]:
                    raise ProtocolError('Commitment seems to be manipulated!')

                # Strip it off
                sigma_permutation = sigma_permutation[:-1]

                inv_sigma_permutation = utils.invert(sigma_permutation)

                # Calculate o_i and g_i
                g_i = utils.permute_graph(scrambled_g, inv_sigma_permutation)

                # Check hash bits
                hash_g_i = SHA512.new(g_i.toarray().tostring()).hexdigest()

                if hash_g_i != ot.hash_g_i:
                    has_cheated = True
                    print("Hash of G_i doesn't match")
                else:
                    print("Hash of G_i matches")

                # Calculate mark_i and wmk_key__i
                mark_i = utils.permute_list(scrambled_mark,
                                            inv_sigma_permutation)
                wmk_key_i = utils.permute_list(scrambled_wmk_key,
                                               inv_sigma_permutation)

                # Extract wmk
                wm_sys = XsWMSystem(**wm_params)
                extracted_msg = wm_sys.extract_watermark(o, syn=mark_i[0:len(
                    mark_i) // 3], key=wmk_key_i)

                print("Extracted mark:\n", np.array(extracted_msg))
                print("Mark_i:\n", mark_i)

                if np.array_equal(extracted_msg, mark_i):
                    print("Extracted mark matches")
                else:
                    has_cheated = True
                    print("Marks are not equal")

            else:
                print('------------------------------------------------------')
                print("Case (ii)")
                print('------------------------------------------------------')
                c2_params = self.prover.reveal(Prover.CHOICES[1])
                rho_permutation = self._aes_decrypt_array(ot.c2,
                                                                *c2_params)
                # Check random number at the end of decrypted commitment
                if nonce != rho_permutation[-1]:
                    raise ProtocolError('Commitment seems to be manipulated!')
                rho_permutation = rho_permutation[:-1]

                # Calculate g_i
                g_i = utils.permute_graph(g, rho_permutation)

                # Check hash bits
                hash_g_i = SHA512.new(g_i.toarray().tostring()).hexdigest()

                if hash_g_i != ot.hash_g_i:
                    has_cheated = True
                    print("Hash of G_i doesn't match")
                else:
                    print("Hash of G_i matches")

            i -= 1

        self.success = not has_cheated

        return self.success

    @staticmethod
    def _aes_decrypt_array(ciphered_perm, key, iv, padding_length):
        """Decrypts an AES in CBC-mode encrypted permutation with the given
        parameters.

        :param ciphered_perm: cipher_text, which should be a string
        representing a permutation, where each element is separated by ','
        :param key: the symmetric key used for encryption
        :param iv: the initialisation vector
        :param padding_length: the amount of padding, which has to be removed
        :return: permutation: the plaintext permutation
        """
        aes_obj = AES.new(key, AES.MODE_CBC, iv)
        recovered_plaintext = aes_obj.decrypt(ciphered_perm)

        if padding_length > 0:
            recovered_plaintext = recovered_plaintext[:-padding_length]

        permutation = np.fromstring(recovered_plaintext, dtype=int)

        return permutation

