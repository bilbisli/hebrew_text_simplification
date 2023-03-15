import re


class Preprocessing():
  """ This class is a hebrew preprocess """
  def __init__(self, sentence):
    self.sentence = sentence

  def remove_punctuation(self):
    """ This method removes punctuation marks """
    self.sentence = re.sub(r'[^\w\s]', '', self.sentence)
  
  def remove_english_letters(self):
    self.sentence = re.sub("\n", " ", self.sentence) 
    self.sentence = re.sub("\xa0", " ", self.sentence) 
    self.sentence = re.sub("\r", " ", self.sentence) 
    self.sentence = re.sub("\r", " ", self.sentence) 
    self.sentence = re.sub("  ", " ", self.sentence)
  
  def remove_text_between_parentheses(self):
    """ This method removes text that appears between Parentheses """
    self.sentence = re.sub("[\(\[].*?[\)\]]", "", self.sentence)
  
  @classmethod
  def remove_dot(cls, sentence):
      return re.sub("\.$", "", sentence.strip())

  def preprocess(self):
    """ This method runs all the reqirements for hebrew data preprocessing. """
    self.remove_text_between_parentheses()
    self.remove_english_letters()

    return self.sentence
