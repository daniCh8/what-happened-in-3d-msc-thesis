from DataManipulator import DataManipulator
from misc_utils import id_to_obj, check_id_is_in_objs, id_set
from collections import Counter
import random


class simple_captioner:
    def __init__(self, path):
        self.data_manip = DataManipulator(path)
        self.valid_hex = '0123456789ABCDEF'.__contains__
        self._rels_dict = dict()

        for rs in self.data_manip._relationships['scans']:
            for r in rs['relationships']:
                self._rels_dict[r[2]] = r[3]

    def auxiliar(self, scan, index):
        ambiguities = set([str(j['instance_source']) for x in scan['ambiguity'] for j in x])
        if str(index) in ambiguities:
            return 'One '
        return 'The '
    
    def filter_relationships_to_id(self, scan_ref, i):
        relationships = self.data_manip.find_scan_relationships(scan_ref)['relationships']
        return [r for r in relationships if str(r[0]) == str(i)]

    def relationships_difference(self, rel_arr_1, rel_arr_2):
        rel_set_1 = set(['{}-{}-{}-{}'.format(x[0], x[1], x[2], x[3]) for x in rel_arr_1])
        rel_set_2 = set(['{}-{}-{}-{}'.format(x[0], x[1], x[2], x[3]) for x in rel_arr_2])
        
        diff_1 = [x.split(sep='-') for x in set.difference(rel_set_1, rel_set_2)]
        diff_2 = [x.split(sep='-') for x in set.difference(rel_set_2, rel_set_1)]
        
        new_diff_1 = [[int(d[0]), int(d[1]), int(d[2]), d[3]] for d in diff_1]
        new_diff_2 = [[int(d[0]), int(d[1]), int(d[2]), d[3]] for d in diff_2]
            
        return new_diff_1, new_diff_2
    
    def cleanhex(self, data):
        return ''.join(filter(self.valid_hex, data.upper()))

    def color_text(self, text, hexcode, mode, id):
        """print in a hex defined color"""
        if mode == 'bash':
            print(f'we here bud {text} {mode}')
            hexint = int(self.cleanhex(hexcode), 16)
            return "\x1B[38;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text)
        else:
            return f'<p style="color:{hexcode}">{id} -- {text}</p>'

    def find_repeated_objects(self, scene_objs):
        c = Counter([x['label'] for x in scene_objs['objects']])
        return [x for x in c.keys() if c[x] > 1]
    
    def filter_relationships(self, rels, repeated_objs, objects_dict):
        random.shuffle(rels)
        new_rels = []
        occurrences = set()
        for r in rels:
            if r[1] not in occurrences and objects_dict[r[1]] not in repeated_objs:
                new_rels.append(r)
                occurrences.add(r[1])
        return new_rels

    def compute_objects_dict(self, objects):
        obj_d = {}
        for o in objects:
            obj_d[int(o['id'])] = o['label']
        return obj_d

    def build_rel_phrase(self, rel_id, rels_dict, present=True):
        is_the = [1, 5, 6, 8, 9, 10, 11, 14, 15, 16, 17, 19, 23, 25, 26, 36, 37, 38, 39]
        to_the = [2, 3]
        in_the = [4]
        it_has = [27, 28, 29, 30, 31]
        it_nil = [12, 13, 18, 20, 21, 22, 24, 32, 33, 34, 35, 40]
        rel_label = rels_dict[rel_id]
        
        if rel_id in is_the:
            if rel_id in [36, 37]:
                rel_label = '{} than'.format(rel_label)
            verb = 'is '
            if not present:
                verb = 'was '
            return verb, '{} '.format(rel_label)
        
        if rel_id in to_the:
            verb = 'is '
            if not present:
                verb = 'was '
            return verb, 'to the {} of '.format(rel_label)
        
        if rel_id in in_the:
            verb = 'is '
            if not present:
                verb = 'was '
            return verb, 'in {} of '.format(rel_label)
        
        if rel_id in it_has:
            rel_label = rel_label.replace(' as', '')
            verb = 'has '
            if not present:
                verb = 'had '
            return verb, 'the {} as '.format(rel_label)
        
        return None, '-1'

    def build_obj_rel_desc(self, rel_array, objects_dict, ref_scan, rels_dict, begin='It', present=True):
        s = begin
        group_1 = [5, 6, 2, 3, 4, 1, 14, 15, 16, 17, 19, 23, 25, 26]
        group_2 = [8, 9, 10, 11]
        group_3 = [27, 28, 29, 30, 31, 36, 37, 38, 39]
        repeated_objs = self.find_repeated_objects(self.data_manip.find_scan_objects(ref_scan['reference']))
        rel_a1 = self.filter_relationships([rel for rel in rel_array if rel[2] in group_1], repeated_objs, objects_dict)
        rel_a2 = self.filter_relationships([rel for rel in rel_array if rel[2] in group_2], repeated_objs, objects_dict)
        rel_a3 = self.filter_relationships([rel for rel in rel_array if rel[2] in group_3], repeated_objs, objects_dict)
        
        if len(rel_a1) == 0 and len(rel_a2) == 0 and len(rel_a3) == 0:
            return ''
        
        for i, rel in enumerate(rel_a1):
            verb, phrase = self.build_rel_phrase(rel[2], rels_dict, present=present)
            auxiliar_2 = self.auxiliar(ref_scan, rel[1]).lower()
            if i == 0:
                s += ' {}{}{}{}'.format(verb, phrase, auxiliar_2, objects_dict[rel[1]])
            elif i == len(rel_a1)-1:
                s += ', and {}{}{}'.format(phrase, auxiliar_2, objects_dict[rel[1]])
            else: 
                s += ', {}{}{}'.format(phrase, auxiliar_2, objects_dict[rel[1]])
        if len(rel_a1) > 0:
            s += '.'
        
        for i, rel in enumerate(rel_a2):
            verb, phrase = self.build_rel_phrase(rel[2], rels_dict, present=present)
            auxiliar_2 = self.auxiliar(ref_scan, rel[1]).lower()
            if i == 0:
                first_one = 'It '
                if len(rel_a1) == 0:
                    first_one = ''
                s += ' {}{}{}{}{}'.format(first_one, verb, phrase, auxiliar_2, objects_dict[rel[1]])
            elif i == len(rel_a2)-1:
                s += ', and {}{}{}'.format(phrase, auxiliar_2, objects_dict[rel[1]])
            else: 
                s += ', {}{}{}'.format(phrase, auxiliar_2, objects_dict[rel[1]])
        if len(rel_a2) > 0:
            s += '.'
        
        last_verb = ''
        for i, rel in enumerate(rel_a3):
            verb, phrase = self.build_rel_phrase(rel[2], rels_dict, present=present)
            verb_char = ''
            if verb != last_verb:
                verb_char = '{}'.format(verb)
            last_verb = verb
            auxiliar_2 = self.auxiliar(ref_scan, rel[1]).lower()
            if i == 0:
                first_one = 'It '
                if len(rel_a1) == 0 and len(rel_a2) == 0:
                    first_one = ''
                s += ' {}{}{}{}{}'.format(first_one, verb, phrase, auxiliar_2, objects_dict[rel[1]])
            elif i == len(rel_a3)-1:
                s += ', and {}{}{}{}'.format(verb_char, phrase, auxiliar_2, objects_dict[rel[1]])
            else: 
                s += ', {}{}{}{}'.format(verb_char, phrase, auxiliar_2, objects_dict[rel[1]])
        if len(rel_a3) > 0:
            s += '.'
        
        return s

    def process_object_description(self, state, before_id, now_id, before_ref, now_ref, mode):
        assert state in ['removed', 'moved', 'added']
        # before is the object ID before;
        # now is the object ID now
        
        if state == 'removed':
            before_objs = self.data_manip.find_scan_objects(before_ref)['objects']
            obj = id_to_obj(before_id, before_objs)
            ref_scan = self.data_manip.find_scan_reference(before_ref)
            beginning_s = self.auxiliar(ref_scan, obj['id']) + obj['label'] + ' has been removed.'
            rel_arr = self.filter_relationships_to_id(before_ref, before_id)
            obj_dict = self.compute_objects_dict(before_objs)
            concat_s = self.build_obj_rel_desc(rel_arr, obj_dict, ref_scan, self._rels_dict, present=False)
            full_s = beginning_s
            if len(concat_s) > 0:
                full_s = '{} {}'.format(beginning_s, concat_s)
            colored_s = self.color_text(full_s, obj['ply_color'], mode, obj['id'])
            return colored_s
        
        if state == 'moved':
            rel_to_id_1 = self.filter_relationships_to_id(before_ref, before_id)
            rel_to_id_2 = self.filter_relationships_to_id(now_ref, now_id)
            rel_arr_1, rel_arr_2 = self.relationships_difference(rel_to_id_1, rel_to_id_2)
            
            before_objs = self.data_manip.find_scan_objects(before_ref)['objects']
            obj = id_to_obj(before_id, before_objs)
            ref_scan = self.data_manip.find_scan_reference(before_ref)
            beginning_s = self.auxiliar(ref_scan, obj['id']) + obj['label'] + ' has been moved.'
            obj_dict = self.compute_objects_dict(before_objs)
            concat_s = self.build_obj_rel_desc(rel_arr_1, obj_dict, ref_scan, self._rels_dict, present=False)
            more_s = beginning_s
            if len(concat_s) > 0:
                more_s = '{} {}'.format(beginning_s, concat_s)
            more_s = self.color_text(more_s, obj['ply_color'], mode, obj['id'])
            
            now_objs = self.data_manip.find_scan_objects(now_ref)['objects']
            now_obj = id_to_obj(now_id, now_objs)
            obj_dict = self.compute_objects_dict(now_objs)
            concat_s = self.build_obj_rel_desc(rel_arr_2, obj_dict, ref_scan, self._rels_dict, present=True, begin='Now, it')
            full_s = more_s
            if len(concat_s) > 0:
                full_s = '{} {}'.format(more_s, self.color_text(concat_s, now_obj['ply_color'], mode, obj['id']))
            return full_s
            
        
        if state == 'added':
            now_objs = self.data_manip.find_scan_objects(now_ref)['objects']
            obj = id_to_obj(now_id, now_objs)
            ref_scan = self.data_manip.find_scan_reference(before_ref)
            beginning_s = self.auxiliar(ref_scan, obj['id']) + obj['label'] + ' has been added.'
            rel_arr = self.filter_relationships_to_id(now_ref, now_id)
            obj_dict = self.compute_objects_dict(now_objs)
            concat_s = self.build_obj_rel_desc(rel_arr, obj_dict, ref_scan, self._rels_dict, present=True)
            full_s = beginning_s
            if len(concat_s) > 0:
                full_s = '{} {}'.format(beginning_s, concat_s)
            colored_s = self.color_text(full_s, obj['ply_color'], mode, obj['id'])
            return colored_s

    def join_string(self, str1, str2):
        if str1 == '':
            return str2
        return f'{str1}\n{str2}'

    def describe_changes(self, reference_scan, changed_scan, mode):
        original_objects = self.data_manip.find_scan_objects(reference_scan)['objects']
        changed_objects = self.data_manip.find_scan_objects(changed_scan['reference'])['objects']
        ref_scan = self.data_manip.find_scan_reference(reference_scan)
        final_string = ''

        for j in changed_scan['removed']:
            if check_id_is_in_objs(j, original_objects):
                add_string = self.process_object_description('removed', j, None, reference_scan, None, mode)
                final_string = self.join_string(final_string, add_string)

        for j in changed_scan['rigid']:
            if check_id_is_in_objs(j['instance_reference'], original_objects) and check_id_is_in_objs(j['instance_rescan'], changed_objects):
                add_string = self.process_object_description('moved', j['instance_reference'], j['instance_rescan'], reference_scan, changed_scan['reference'], mode)
                final_string = self.join_string(final_string, add_string)

        for j in set.difference(id_set(changed_objects), id_set(original_objects)):
            if check_id_is_in_objs(j, changed_objects):
                add_string = self.process_object_description('added', None, j, reference_scan, changed_scan['reference'], mode)
                final_string = self.join_string(final_string, add_string)
        
        return final_string

    def create_changes_html(self, ref, chg, save_path):
        f = open(save_path, 'w')
        html = self.describe_changes(ref, chg, 'html')
        f.write(html)
        f.close()
    