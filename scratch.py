from opyenxes.model.XLog import XLog
from opyenxes.data_in.XUniversalParser import XUniversalParser
from opyenxes.classification.XEventAttributeClassifier import XEventAttributeClassifier



path = "logs/BPI Challenge 2011.xes"
# Parse Logs
with open(path) as log_file:
    rlog = XUniversalParser().parse(log_file)

log = rlog[0]
raw_event = log[0][0]
raw_attributes = raw_event.get_attributes()

for attribute in raw_attributes:
    attribute = raw_attributes[attribute]
    print(attribute.get_key(), attribute.get_value())
