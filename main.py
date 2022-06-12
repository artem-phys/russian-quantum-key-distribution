import json
from qkdsimulationprotocol import QKDSimulationProtocol


def main():
    with open('input.json') as fin:
        input_data = json.load(fin)

    initial_key_length = input_data['initial_key_length']
    eavesdropping_rate = input_data['eavesdropping_rate']
    error_estimation_sampling_rate = input_data['error_estimation_sampling_rate']
    error_reconciliation_efficiency = input_data['error_reconciliation_efficiency']

    qkd_protocol = QKDSimulationProtocol(initial_key_length, eavesdropping_rate, error_estimation_sampling_rate, error_reconciliation_efficiency)
    qkd_protocol.run()
    qkd_protocol.print_summary()

    with open('output.json', 'w') as fout:
        json.dump(qkd_protocol.output(), fout)


if __name__ == '__main__':
    main()

