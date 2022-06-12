import json
import main


def l_sec_minmax(input_data):
    eavesdropping_rate = input_data['eavesdropping_rate']

    if eavesdropping_rate == 0:
        l_min = 3800
        l_max = 4200
    elif eavesdropping_rate == 0.12:
        l_min = 1936
        l_max = 2683
    elif eavesdropping_rate == 1:
        l_min = 0
        l_max = 0

    return [l_min, l_max]


def program_verifier(input_data, output_data):
    print()
    print('ALGORITHM VERIFIER OUTPUT')
    l_sec = output_data['l_sec']
    alice_final_key = output_data['alice_final_key']
    bob_final_key = output_data['bob_final_key']

    l_min, l_max = l_sec_minmax(input_data)
    print(f'l_min = {l_min}')
    print(f'l_max = {l_max}')

    print(f'Length verification status: {l_min <= l_sec <= l_max}')

    if l_sec == 0:
        print(f'Similarity verification is not necessary')
    else:
        print(f'Similarity verification status: {alice_final_key == bob_final_key}')


if __name__ == '__main__':
    main.main()  # execute the program

    with open('input.json') as fin:
        input_json = json.load(fin)

    with open('output.json') as fout:
        output_json = json.load(fout)

    program_verifier(input_json, output_json)
