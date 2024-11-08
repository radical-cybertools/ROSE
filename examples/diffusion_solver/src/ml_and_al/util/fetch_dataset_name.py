def fetch_dataset_name(idx):
    name_number_dict = {18: 'EighteenSrcsRdm',
                        14: 'FourteenSrcsRdm',
                        13: 'ThirteenSrcsRdm',
                        19: 'NineteenSrcsRdm',
                        12: 'TwelveSrcsRdm',
                        17: 'SeventeenSrcsRdm',
                        20: 'TwentySrcsRdm',
                        15: 'FifteenSrcsRdm',
                        16: 'SixteenSrcsRdm',
                        11: 'ElevenSrcsRdm',
                        1 : '1SourcesRdm',
                        # 2 : '2SourcesRdm',
                        2: 'TwoSourcesRdm',
                        3 : '3SourcesRdm',
                        4 : '4SourcesRdm',
                        5 : '5SourcesRdm',
                        6 : '6SourcesRdm',
                        7 : '7SourcesRdm',
                        8 : '8SourcesRdm',
                        9 : '9SourcesRdm',
                        10: '10SourcesRdm'}
    if idx not in name_number_dict.keys():
        return None
    return name_number_dict[idx]

def get_number_sources_from_dataset_name(name):
    name_number_dict = {'EighteenSrcsRdm' : 18,
                        'FourteenSrcsRdm' : 14,
                        'ThirteenSrcsRdm' : 13,
                        'NineteenSrcsRdm' : 19,
                        'TwelveSrcsRdm'   : 12,
                        'SeventeenSrcsRdm': 17,
                        'TwentySrcsRdm'   : 20,
                        'FifteenSrcsRdm'  : 15,
                        'SixteenSrcsRdm'  : 16,
                        'ElevenSrcsRdm'   : 11,
                        '1SourcesRdm'     : 1,
                        # '2SourcesRdm'     : 2,
                        'TwoSourcesRdm'   : 2,
                        '3SourcesRdm'     : 3,
                        '4SourcesRdm'     : 4,
                        '5SourcesRdm'     : 5,
                        '6SourcesRdm'     : 6,
                        '7SourcesRdm'     : 7,
                        '8SourcesRdm'     : 8,
                        '9SourcesRdm'     : 9,
                        '10SourcesRdm'    : 10,
                        '20SourcesRdm'    : 10}
    return name_number_dict[name]


def get_Nplus_Nminus(set_name):
    N = get_number_sources_from_dataset_name(set_name)
    Nplus = N + 1
    Nminus = N - 1
    name_plus = fetch_dataset_name(Nplus)
    name_minus = fetch_dataset_name(Nminus)

    return name_plus, name_minus
