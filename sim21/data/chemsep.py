import math
import os
import sqlite3
from dataclasses import dataclass
from importlib import resources
import numpy as np
from sim21.data.eqn import eval_eqn, eval_eqn_int_over_t, eval_eqn_int
from sim21.data.chemsep_consts import GAS_CONSTANT

with resources.path('sim21.data', 'chemsepv812.db') as db_path:
    _CHEMSEP_DATABASE_CON = sqlite3.connect(db_path)
    _CHEMSEP_DATABASE_CON.row_factory = sqlite3.Row
    _CHEMSEP_DATABASE_CUR = _CHEMSEP_DATABASE_CON.cursor()


def read_chemsep_xml_to_dict(source):
    """
    Extract chemsep resources from source the file and return the dictionary
    containing the database. The source file must be the xml resources dump
    from the Chemsep database.

    source: string with the path to the chemsep resources file.
    """
    from xml.dom.minidom import parse
    dom1 = parse(source)
    compounds = dom1.childNodes[0]
    all_prop_keys = []
    # all_comps = []
    prop_count = {}
    compound_count = 0
    all_component_data = {}
    # logger.debug("Loading ChemSep Database...")
    print("Loading ChemSep Database...")
    for comp in compounds.childNodes:
        comp_dict = {}

        # if have a blank, ignore
        if comp.nodeType == comp.TEXT_NODE:
            continue

        for prop in comp.childNodes:
            if prop.nodeType == prop.TEXT_NODE:
                continue

            attr = prop.attributes
            if attr is None:
                continue
            for key, value in list(attr.items()):
                if key == 'name':
                    continue
                index = prop.nodeName + '_' + key
                # print('\t', key, '[', value, ']', index)
                if index not in prop_count:
                    prop_count[index.lower()] = [value]
                else:
                    prop_count[index.lower()].append(value)

                new_key_value = index.lower()
                if new_key_value not in comp_dict:
                    comp_dict[new_key_value] = [value]
                else:
                    comp_dict[new_key_value].append(value)

            v = prop.childNodes
            for vsub in v:
                if vsub.nodeType == vsub.TEXT_NODE:
                    continue
                # print '\t\t', vsub.nodeName
                vsub_attr = vsub.attributes
                if vsub_attr is None:
                    continue
                for vsub_key, vsub_value in list(vsub_attr.items()):
                    index = prop.nodeName + '_' + vsub.nodeName + '_' + vsub_key
                    # print('\t\t\t', vsub_key, '[', vsub_value, ']', index)

                    if index not in prop_count:
                        prop_count[index.lower()] = [vsub_value]
                    else:
                        prop_count[index.lower()].append(vsub_value)

                    new_key_value = index.lower()
                    if new_key_value not in comp_dict:
                        comp_dict[new_key_value] = [vsub_value]
                    else:
                        comp_dict[new_key_value].append(vsub_value)

        for key in list(comp_dict.keys()):
            if key not in all_prop_keys:
                all_prop_keys.append(key)

        # all_comps.append(comp_dict)
        identifier = comp_dict['compoundid_value'][0].upper()
        all_component_data[identifier] = comp_dict
        print('#', compound_count, identifier)
        compound_count += 1

    # logger.debug(compound_count, "components are available.")
    print(compound_count, "components are available.")

    # Transform all resources
    # print('Internalizing components...')
    intern_comp_data = {}

    for k, v in all_component_data.items():
        intern_comp_data[k] = v

    return intern_comp_data


def create_chemsep_database(xml_source_path, output_file_path):
    # Read all the raw data from xml_source_path
    raw_data = read_chemsep_xml_to_dict(xml_source_path)
    raw_data_step2 = {}
    all_possible_keys = set()

    for key, value in raw_data.items():
        raw_data_step2[key] = {sub_key: ','.join(sub_value) for sub_key, sub_value in value.items()}
        all_possible_keys = all_possible_keys.union([sub_key for sub_key in value.keys()])

    # Convert to possible keys to a list
    all_possible_keys = list(sorted(all_possible_keys))

    # Now create the database
    # Delete the database
    try:
        os.remove(output_file_path)
    except FileNotFoundError:
        pass

    conn = sqlite3.connect(output_file_path)
    c = conn.cursor()

    # Create the table with the keys
    keys_entries = ',\n'.join([i + ' text' for i in all_possible_keys])
    prefix = 'CREATE TABLE IF NOT EXISTS pure (identifier integer PRIMARY KEY, \n'
    pure_table_create_script = prefix + keys_entries + ');\n'
    c.execute(pure_table_create_script)

    # Create the template to add records to the database
    col_names = all_possible_keys[:]
    col_names_text = ','.join(col_names)
    col_values_text = ','.join(['?'] * len(col_names))
    script_text = 'INSERT INTO pure(' + col_names_text + ') VALUES(' + col_values_text + ')'

    # Create a list of all the records
    records = []
    for key, value in raw_data_step2.items():
        # c = conn.cursor()
        col_values = [value.get(i, '') for i in col_names]
        # do a self check
        for k, v in zip(col_names, col_values):
            if k in value:
                assert value[k] == v

        records.append(col_values)

    # Add items to the database
    c.executemany(script_text, records)

    # Commit changes and we'r edone
    conn.commit()
    conn.close()


def conv_number(v, typeofvalue=float, fallback_value=0):
    try:
        return typeofvalue(v)
    except ValueError:
        return fallback_value


@dataclass
class ChemsepPure:
    identifier: str
    acen_fact: float              # Unit less
    crit_temp: float              # K
    crit_press: float             # Pa
    crit_compress_fact: float     # Unit less
    crit_vol_mole: float          # m3/kmol
    ig_enthalpy_form_mole: float  # J/kmol
    ig_entropy_form_mole: float   # J/kmol-K
    ig_gibbs_form_mole: float     # J/kmol
    ig_heat_cap_mole_coeffs: np.ndarray   # J/kmol
    ig_temp_ref: float            # K
    ig_press_ref: float           # Pa
    liq_visc_coeffs: np.ndarray
    vap_visc_coeffs: np.ndarray

    def __init__(self, name):
        cur = _CHEMSEP_DATABASE_CUR
        cur.execute('SELECT * FROM pure WHERE UPPER(compoundid_value) LIKE ?', [name.upper()])
        results = cur.fetchall()
        assert len(results) > 0
        col_ids = results[0].keys()
        row_data = results[0]
        comp_data = {k: v for k, v in zip(col_ids, row_data)}
        # print('\n'.join(sorted(comp_data.keys())))
        # print(comp_data['liquidvolumeatnormalboilingpoint_units'])
        # print(comp_data['liquidvolumeatnormalboilingpoint_value'])
        self.identifier = comp_data['compoundid_value'].upper()
        self.acen_fact = conv_number(comp_data['acentricityfactor_value'])
        self.crit_temp = conv_number(comp_data['criticaltemperature_value'])
        self.crit_press = conv_number(comp_data['criticalpressure_value'])
        self.crit_compress_fact = conv_number(comp_data['criticalcompressibility_value'])
        self.crit_vol_mole = conv_number(comp_data['criticalvolume_value'])
        self.mw = conv_number(comp_data['molecularweight_value'])
        self.ig_enthalpy_form_mole = enthalpy = conv_number(comp_data['heatofformation_value'])
        self.ig_gibbs_form_mole = gibbs = conv_number(comp_data['gibbsenergyofformation_value'])
        self.ig_cp_mole_coeffs = np.array([conv_number(comp_data['rppheatcapacitycp_eqno_value']),
                                           conv_number(comp_data['rppheatcapacitycp_tmin_value']),
                                           conv_number(comp_data['rppheatcapacitycp_tmax_value']),
                                           conv_number(comp_data['rppheatcapacitycp_a_value']),
                                           conv_number(comp_data['rppheatcapacitycp_b_value']),
                                           conv_number(comp_data['rppheatcapacitycp_c_value']),
                                           conv_number(comp_data['rppheatcapacitycp_d_value']),
                                           conv_number(comp_data['rppheatcapacitycp_e_value']),
                                           0.0])
        self.liq_visc_coeffs = np.array([conv_number(comp_data['liquidviscosity_eqno_value']),
                                         conv_number(comp_data['liquidviscosity_tmin_value']),
                                         conv_number(comp_data['liquidviscosity_tmax_value']),
                                         conv_number(comp_data['liquidviscosity_a_value']),
                                         conv_number(comp_data['liquidviscosity_b_value']),
                                         conv_number(comp_data['liquidviscosity_c_value']),
                                         conv_number(comp_data['liquidviscosity_d_value']),
                                         conv_number(comp_data['liquidviscosity_e_value']),
                                         0.0])

        self.vap_visc_coeffs = np.array([conv_number(comp_data['vaporviscosity_eqno_value']),
                                         conv_number(comp_data['vaporviscosity_tmin_value']),
                                         conv_number(comp_data['vaporviscosity_tmax_value']),
                                         conv_number(comp_data['vaporviscosity_a_value']),
                                         conv_number(comp_data['vaporviscosity_b_value']),
                                         conv_number(comp_data['vaporviscosity_c_value']),
                                         conv_number(comp_data['vaporviscosity_d_value']),
                                         conv_number(comp_data['vaporviscosity_e_value']),
                                         0.0])

        self.std_liq_vol_mole = conv_number(comp_data['liquidvolumeatnormalboilingpoint_value'])
        self.ig_temp_ref = 298.15
        self.ig_press_ref = 101325.0
        self.ig_entropy_form_mole = (gibbs - enthalpy)/(-self.ig_temp_ref)

    def ig_heat_cap_mole(self, temp):
        return eval_eqn(self.ig_cp_mole_coeffs, temp, self.crit_temp)

    def ig_enthalpy_mole(self, temp):
        return eval_eqn_int(self.ig_cp_mole_coeffs, temp, self.ig_temp_ref, self.ig_enthalpy_form_mole)

    def ig_entropy_mole(self, temp, press):
        p1 = eval_eqn_int_over_t(self.ig_cp_mole_coeffs, temp, self.ig_temp_ref, self.ig_entropy_form_mole)
        return p1 - GAS_CONSTANT * math.log(press / self.ig_press_ref)

    def ig_gibbs_mole(self, temp, press):
        return self.ig_enthalpy_mole(temp) - self.ig_entropy_mole(temp, press)

    def liq_visc(self, temp):
        return eval_eqn(self.liq_visc_coeffs, temp, self.crit_temp)

    def vap_visc(self, temp):
        return eval_eqn(self.vap_visc_coeffs, temp, self.crit_temp)


def pure(names):
    if isinstance(names, str):
        return ChemsepPure(names)

    assert not isinstance(names, str)
    return [ChemsepPure(i) for i in names]


def available():
    cur = _CHEMSEP_DATABASE_CUR
    cur.execute('SELECT UPPER(compoundid_value) FROM pure')
    results = cur.fetchall()
    return [i[0] for i in results]
