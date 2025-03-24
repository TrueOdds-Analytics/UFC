"""
Category encoding classes for consistent feature processing
"""
import pickle


class CategoryEncoder:
    """Ensures consistent categorical encoding across different datasets"""

    def __init__(self):
        self.category_mappings = {}
        self.initialized = False

    def fit(self, data):
        """Learn category mappings from reference data"""
        category_columns = [col for col in data.columns if col.endswith(('fight_1', 'fight_2', 'fight_3'))]

        for col in category_columns:
            # Create mapping from unique values to integer codes
            unique_values = data[col].dropna().unique()
            self.category_mappings[col] = {val: i for i, val in enumerate(unique_values)}

        self.initialized = True
        return self

    def transform(self, data):
        """Apply consistent categorical mappings"""
        if not self.initialized:
            raise ValueError("CategoryEncoder must be fit before transform")

        # Create a copy to avoid modifying the original
        data_copy = data.copy()

        # Process each categorical column
        for col, mapping in self.category_mappings.items():
            if col in data_copy.columns:
                # Apply mapping, with -1 for unknown values
                data_copy[col] = data_copy[col].map(mapping).fillna(-1).astype('int32')

        return data_copy

    def fit_transform(self, data):
        """Fit and transform in one step"""
        self.fit(data)
        return self.transform(data)

    def save(self, filepath):
        """Save encoder to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.category_mappings, f)

    @classmethod
    def load(cls, filepath):
        """Load encoder from disk"""
        encoder = cls()
        with open(filepath, 'rb') as f:
            encoder.category_mappings = pickle.load(f)
        encoder.initialized = True
        return encoder