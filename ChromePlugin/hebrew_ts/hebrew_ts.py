import sys
import os.path
import argparse

from hebrew_ts.models import mask_model, ner_model, ner_tokenizer
from hebrew_ts.ts_methods import simplify_words, generate_summary


def argparse_wrapper(func):
    def feed_func(args):
        return func(**vars(args))
    return feed_func

@argparse_wrapper
def hebrew_ts_cli(*args, **kwargs):
    return text_simplification_pipeline(*args, **kwargs)

def text_simplification_pipeline(text, score_threshold=0.32, neighbours_threshold=0.7, word_sub=True, sentence_filter=True, new_line_tok='\n', new_line_model_token='<NL>'):
    if os.path.isfile(text):
        print('in')
        with open(text, 'r', encoding='utf-8') as f:
            text = f.read()
            print(text)
    simp_list = []
    summ_list = []
    paragraphs = [p for p in text.split(new_line_tok) if p != '' and not p.isspace()]
    for paragraph in paragraphs:
        if word_sub:
            simp = simplify_words(paragraph, tokenizer=ner_tokenizer, ner_model=ner_model, mask_model=mask_model, score_threshold=score_threshold, neighbours_threshold=neighbours_threshold, new_line_model_token=new_line_model_token)
        else:
            simp = paragraph
        if sentence_filter:
            summ = generate_summary(simp, top_n_func=None, new_line_tok=new_line_tok, visualize=False)
        else:
            summ = paragraph
        simp_list.append(simp)
        summ_list.append(summ)

    simp_text = new_line_tok.join(simp_list)
    summ_text = new_line_tok.join(summ_list)
    
    return summ_text

text1 ="""ג המולד הוא חג המקובל כמעט בכל זרמי הנצרות המציין את הולדת ישו. על פי הברית החדשה נולד ישו למרים הבתולה בבית לחם במקום בו מצויה כיום כנסיית המולד. הנוצרים מאמינים כי לידתו של ישו מהווה הגשמה של נבואות התנ"ך על בואו של משיח מבית דוד, אשר יגאל את העולם מחטאיו ויגשר על הפער שבין האל ובין בני האדם. אין הסכמה בקרב הכנסיות ובקרב היסטוריונים על הכרונולוגיה המדויקת של לידת ישו, אך מאז המאה הרביעית כמעט כל הכנסיות חוגגות אותו ב-25 בדצמבר (כך גם הכנסיות המזרחיות המשתמשות בלוח היוליאני לצרכים דתיים, אם כי בשל השימוש האזרחי במקבילו הגרגוריאני הן מציינות אותו לכאורה ב-7 בינואר)."""
# with open('text_file.txt', 'w+', encoding='utf-8') as f:
#     f.write(text1)

def main(parser=None):
    parser = parser or argparse.ArgumentParser(prog='hebrew_ts', description='Simplifies complex Hebrew text',)
    parser.add_argument('text', type=str, metavar='TEXT', help=f'Text (or file of text) to simplify. example1: text_file.txt\nexample2:{text1}')
    parser.add_argument('--word_sub', dest='word_sub', action='store_true', help='Do word substitution. example: --word_sub', default=True)
    parser.add_argument('--no-word_sub', dest='word_sub', action='store_false', help='Do not do word substitution. example: --no-word_sub')
    parser.add_argument('--sentence_filter', dest='sentence_filter', action='store_true', help='Do sentence filtering. example: --sentence_filter', default=True)
    parser.add_argument('--no-sentence_filter', dest='sentence_filter', action='store_false', help='Do not do sentence filtering. example: --no-sentence_filter')
    parser.add_argument('--score_threshold', dest='score_threshold', default=0.32, type=float, help='Threshold for word substitution. example: -score_threshold=0.32')
    parser.add_argument('--neighb_thresh', dest='neighbours_threshold', default=0.7, type=float, help='Do not do word substitution. example: -neighb_thresh=0.7')
    args = parser.parse_args()
    res = hebrew_ts_cli(args)
    print(res)

if __name__ == '__main__':
    sys.stdin.reconfigure(encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    main()
