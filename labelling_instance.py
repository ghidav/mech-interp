from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
import copy

class LabellingInstance:
    def get_positive_labels(self):
        if not self._finished_labelling:
            print("You haven't finished labelling yet")
        else:
            return copy.deepcopy(self._positive_labels)

    def get_negative_labels(self):
        if not self._finished_labelling:
            print("You haven't finished labelling yet")
        else:
            return copy.deepcopy(self._positive_labels)

    def __init__(self, phrases_list, positive_labels_name, negative_labels_name):
        self._original_phrases_length = len(phrases_list)
        self._positive_labels_name = positive_labels_name
        self._negative_labels_name = negative_labels_name
        self._finished_labelling = False
        self._phrases = copy.deepcopy(phrases_list)
        self._positive_labels = {}
        self._negative_labels = {}
        self._phrase_hash = -1
        self._previous_phrase = None
        self._last_positive_hash = -1
        self._last_negative_hash = -1

        # Create buttons
        self._positive_button = widgets.Button(description=self._positive_labels_name, button_style="success")
        self._negative_button = widgets.Button(description=self._negative_labels_name, button_style="danger")

        # Set button click handlers
        self._positive_button.on_click(self._on_button_click)
        self._negative_button.on_click(self._on_button_click)

        # Initial display
        self._display_next_phrase()

    def _share_same_prompt(self, phrase_1, phrase_2, min_prompt_length=20):
        if phrase_1 is None or phrase_2 is None or len(phrase_1) < 20 or len(phrase_2) < 20: return False
        return phrase_1[:20] == phrase_2[:20]

    def _on_button_click(self, b):
        clear_output(wait=True)  # Clear the output to update the display
        if b.description == self._positive_labels_name:
            if self._last_positive_hash == self._phrase_hash:
                self._positive_labels[self._phrase_hash].append(current_phrase)
            else:
                self._positive_labels[self._phrase_hash] = [current_phrase]
                self._last_positive_hash = self._phrase_hash
        else:
            if self._last_negative_hash == self._phrase_hash:
                self._negative_labels[self._phrase_hash].append(current_phrase)
            else:
                self._negative_labels[self._phrase_hash] = [current_phrase]
                self._last_negative_hash = self._phrase_hash
        self._display_next_phrase()

    def _display_next_phrase(self):
        global current_phrase
        if self._phrases:
            current_phrase = self._phrases.pop(0)
            if not self._share_same_prompt(current_phrase, self._previous_phrase):
                self._phrase_hash = self._phrase_hash + 1
                self._previous_phrase = current_phrase
            display(HTML(f"<h3>{current_phrase}</h3>"))
            display(self._positive_button, self._negative_button)
        else:
            print("No more phrases to label.")
            self._finished_labelling = True
            positive_list = []
            for key in self._positive_labels.keys():
                positive_list = positive_list + self._positive_labels[key]

            negative_list = []
            for key in self._negative_labels.keys():
                negative_list = negative_list + self._negative_labels[key]

            assert len(set(positive_list + negative_list)) == len(positive_list + negative_list)
            assert len(set(positive_list + negative_list)) == self._original_phrases_length
