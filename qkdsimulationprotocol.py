# CLASS QKD
import numpy as np
import math
import random

from qiskit import QuantumCircuit, ClassicalRegister, Aer, transpile
from scipy.linalg import toeplitz


# SUPPLEMENTARY FUNCTIONS


def error_counts(alice_key, bob_key):
    return sum([alice_key[i] ^ bob_key[i] for i in range(len(alice_key))])


# Shannon's entropy
def h(p):
    if p == 0 or p == 1:
        return 0
    else:
        return - p * np.log2(p) - (1 - p) * np.log2(1 - p)


class QKDSimulationProtocol:
    """
    QKD simulation protocol entity.
    Keep all variables and intermediate steps of the protocol.
    Steps are implemented as methods.
    Start a full run via "QKD_run" method
    """

    def __init__(self, initial_key_length_=10000, eavesdropping_rate_=0, error_estimation_sampling_rate_=0.2, error_reconciliation_efficiency_=1.4):
        """Constructor from user input """

        self.initial_key_length = initial_key_length_
        self.eavesdropping_rate = eavesdropping_rate_
        self.error_estimation_sampling_rate = error_estimation_sampling_rate_
        self.error_reconciliation_efficiency = error_reconciliation_efficiency_

        self.qc = None
        self.memory = None
        self.counts = None
        self.alice_sifted_key = None
        self.bob_sifted_key = None
        self.l_sift = None
        self.real_qber = None
        self.n_error_estimation_bits = None
        self.estimated_qber = None
        self.n_sifted_after_ee = None
        self.bob_key_after_ee = None
        self.alice_key_after_ee = None
        self.reconciliation_qber = None
        self.alice_reconciled_key = None
        self.bob_reconciled_key = None
        self.leaked_bits = None
        self.error_correction_success_flag = None
        self.l_ver = None
        self.l_sec = None
        self.alice_final_key = None
        self.bob_final_key = None

    def init_circuit(self):
        """Qiskit quantum circuit creation"""
        qc = QuantumCircuit(7, 4)

        # Random bit and basis generation by Alice
        qc.h(1)
        qc.h(2)
        qc.cnot(1, 0)
        qc.ch(2, 0)
        qc.measure(1, 3)
        qc.measure(2, 2)

        # Quantum communication channel - Eve (Resend attack)
        qc.barrier()

        # Eve's classical registers for storing bases selections and measurements
        qc.add_register(ClassicalRegister(2, name='c1'))
        qc.add_register(ClassicalRegister(1, name='c2'))

        # Eve's measurement
        eavesdropping_theta = 2 * math.asin(math.sqrt(self.eavesdropping_rate))  # from 0 to pi
        qc.r(eavesdropping_theta, 0, 6)
        qc.cswap(6, 0, 5)
        qc.h(4)
        qc.ch(4, 5)
        qc.measure(4, qc.cregs[1][0])
        qc.measure(5, qc.cregs[2][0])

        # Eve's resending
        qc.initialize([1, 0], 5)
        qc.x(5).c_if(qc.cregs[2], 1)
        qc.h(5).c_if(qc.cregs[1], 1)
        qc.cswap(6, 0, 5)
        qc.measure(6, qc.cregs[1][1])

        # Random basis selection and measurement by Bob
        qc.barrier()
        qc.h(3)
        qc.ch(3, 0)
        qc.measure(3, 1)
        qc.measure(0, 0)

        # Save quantum circuit
        self.qc = qc

    def simulate_circuit(self):
        """Simulation via Aer backend """
        backend = Aer.get_backend('aer_simulator')
        n_shots = self.initial_key_length
        job = backend.run(transpile(self.qc, backend), shots=n_shots, memory=True)

        self.counts = job.result().get_counts()
        self.memory = job.result().get_memory()

    def readout_interpretation(self):
        """Help for interpretation of qiskit simulation readouts"""
        eve_actions = ['Eve ignored that bit', 'Eve measured and resent that bit']
        bases_signs = ['+', '🞩']
        for readout in dict(sorted(self.counts.items(), key=lambda item: item[1], reverse=True)).keys():
            # decomposition of a readout (e.g. '0 11 0110')
            eve_bit = int(readout[0])
            eve_basis = int(readout[3])
            eve_presence = int(readout[2])

            alice_bit = int(readout[5])
            alice_basis = int(readout[6])
            bob_basis = int(readout[7])
            bob_bit = int(readout[8])

            readout_summary = (
                    readout + '\n' +
                    f'Alice bit {alice_bit}' + '\n' +
                    f'Alice basis {bases_signs[alice_basis]}' + '\n' +
                    f'Eve basis {bases_signs[eve_basis]}' + '\n' +
                    f'{eve_actions[eve_presence]}' + '\n' +
                    f'Eve bit {eve_bit}' + '\n' +
                    f'Bob basis {bases_signs[bob_basis]}' + '\n' +
                    f'Bob bit {bob_bit}' + '\n'
            )
            print(readout_summary)

    def sifting(self):
        """Creation of sifted keys from simulation job memory readouts"""
        self.alice_sifted_key = []
        self.bob_sifted_key = []

        for shot_readout in self.memory:
            if shot_readout[6] == shot_readout[7]:
                self.alice_sifted_key.append(int(shot_readout[5]))
                self.bob_sifted_key.append(int(shot_readout[8]))

        self.l_sift = len(self.alice_sifted_key)

    def calculate_qber(self):
        """Real QBER calculation """
        self.real_qber = error_counts(self.alice_sifted_key,
                                      self.bob_sifted_key) / self.l_sift if self.l_sift != 0 else 0

    def error_estimation_protocol(self):
        """Estimated QBER calculation and deleting all the disclosed bits"""
        self.n_error_estimation_bits = round(self.error_estimation_sampling_rate * self.l_sift)
        indices_for_ee = random.sample(range(self.l_sift), self.n_error_estimation_bits)

        self.estimated_qber = error_counts([self.alice_sifted_key[j] for j in indices_for_ee],
                                           [self.bob_sifted_key[j] for j in
                                            indices_for_ee]) / self.n_error_estimation_bits if self.n_error_estimation_bits != 0 else 0.1

        self.alice_key_after_ee = [self.alice_sifted_key[i] for i in range(self.l_sift) if i not in indices_for_ee]
        self.bob_key_after_ee = [self.bob_sifted_key[i] for i in range(self.l_sift) if i not in indices_for_ee]

        self.n_sifted_after_ee = len(self.alice_key_after_ee)

    def error_reconciliation_protocol(self):

        self.alice_reconciled_key = self.alice_key_after_ee
        self.bob_reconciled_key = self.alice_reconciled_key

        self.reconciliation_qber = error_counts(self.bob_key_after_ee, self.bob_reconciled_key) / self.n_sifted_after_ee

        self.leaked_bits = math.ceil(self.error_reconciliation_efficiency * h(self.reconciliation_qber) * self.n_sifted_after_ee)

    def error_reconciliation_verification(self):
        """Checking if keys are the same after error reconciliation"""

        if hash(str(self.alice_reconciled_key)) == hash(str(self.bob_reconciled_key)):
            self.error_correction_success_flag = 1
            self.l_ver = len(self.alice_reconciled_key)
        else:
            self.error_correction_success_flag = 0
            self.l_ver = 0

    def privacy_amplification(self):
        """Universal hashing scheme based on Toeplitz matrices"""

        self.l_sec = max(self.l_ver - math.ceil(self.l_ver * h(self.reconciliation_qber) + self.leaked_bits), 0)

        # Protocol failure. Zero key material was generated during this run
        if self.l_sec == 0 or self.error_correction_success_flag == 0:
            self.alice_final_key = []
            self.bob_final_key = []
            self.l_sec = 0
        # Protocol success
        else:
            bitstring1 = np.random.randint(2, size=self.l_sec)
            bitstring2 = np.random.randint(2, size=self.l_ver)
            tm = toeplitz(bitstring1, bitstring2)

            self.alice_final_key = np.dot(tm, self.alice_reconciled_key) % 2
            self.bob_final_key = np.dot(tm, self.bob_reconciled_key) % 2
            self.l_sec = len(self.bob_final_key)

    def run(self):
        self.init_circuit()
        self.simulate_circuit()
        self.sifting()
        self.calculate_qber()
        self.error_estimation_protocol()
        self.error_reconciliation_protocol()
        self.error_reconciliation_verification()
        self.privacy_amplification()

    def print_summary(self):
        precision = 3
        print(f'USER INPUT')
        print(f'Initial key length : {self.initial_key_length}')
        print(f'Eavesdropping rate : {self.eavesdropping_rate:.{precision}f}')
        print(f'Error estimation sampling rate : {self.error_estimation_sampling_rate}')
        print(f'Error reconciliation efficiency = {self.error_reconciliation_efficiency}')
        print()

        print('BB84 RUN')
        print(f'Number of transmitted qubits: l_raw = {self.initial_key_length}')
        print(f'Eve measured and resent {len([i for i in range(self.initial_key_length) if int(self.memory[i][2]) == 1])} bits')
        print()

        print('SIFTING')
        print(f'The sifted key length: l_sift = {self.l_sift}')
        print(f'Real QBER = {self.real_qber:.{precision}}')
        print()

        print('ERROR ESTIMATION')
        print(f'Bits disclosed for error estimation = {self.n_error_estimation_bits}')
        print(f'Estimated QBER = {self.estimated_qber:.{precision}}')
        print(f'Key length after error estimation = {self.n_sifted_after_ee}')
        print()

        print('ERROR RECONCILIATION')
        print(f'Error reconciliation status: {"success" if self.error_correction_success_flag else "failed"}')
        print(f'Reconciliation QBER = {self.reconciliation_qber:.{precision}}')
        print(f'Shannon bound for leaked bits: n_shannon = {int(h(self.reconciliation_qber) * self.n_sifted_after_ee)}')
        print(f'Error reconciliation efficiency = {self.error_reconciliation_efficiency}')
        print(f'Information leakage during error reconciliation n_leaked = {self.leaked_bits}')
        print()

        print('PRIVACY AMPLIFICATION')
        print(f'The length of the verified key before running privacy amplification: l_ver = {self.l_ver}')
        print(f'The final key length: l_sec = {self.l_sec}')
        print(
            f'Alice and Bob\'s final keys are the same : {(self.alice_final_key == self.bob_final_key).all() if self.l_sec != 0 else True}')

    def output(self):
        alice_final_key_str = ''.join(map(str, self.alice_final_key))
        bob_final_key_str = ''.join(map(str, self.bob_final_key))

        return {'l_sec': self.l_sec, 'alice_final_key': alice_final_key_str, 'bob_final_key': bob_final_key_str}
