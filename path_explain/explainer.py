"""
Contains the base class for the two explainer objects.
"""

class Explainer():
    """
    A superclass for all explainer objects.
    This (somewhat) matches the SHAP
    package in terms of API.
    """

    def attributions(self,
                     inputs,
                     baseline,
                     batch_size=50,
                     num_samples=100,
                     use_expectation=True,
                     output_indices=None,
                     verbose=False):
        """
        A function that returns the path attributions for the
        given inputs.
        """
        raise Exception("Attributions have not been implemented " + \
                    "for this class. Likely, you have imported " + \
                    "the wrong class from this package.")

    def interactions(self, inputs,
                     baseline,
                     batch_size=50,
                     num_samples=100,
                     use_expectation=True,
                     output_indices=None,
                     verbose=False,
                     interaction_index=None):
        """
        A function that returns the path interactions for the
        given inputs.
        """
        raise Exception("Interactions have not been implemented " + \
                    "for this class. Likely, you have imported " + \
                    "the wrong class from this package.")
