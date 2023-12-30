import unittest

class TestBigQuery(unittest.TestCase):

    def test_get_contentID_keyName(self):
        self.assertEqual(get_contentID_keyName("pmr_pm_plan"), (36, "pm_id", "merge_pm_plan"))
        self.assertEqual(get_contentID_keyName("pmr_pm_item"), (37, "pm_item_id", "merge_pm_item"))
        self.assertEqual(get_contentID_keyName("pmr_project"), (7, "project_id", "merge_project"))
        self.assertEqual(get_contentID_keyName("pmr_inventory"), (14, "inventory_id", "merge_inventory"))
        self.assertEqual(get_contentID_keyName("invalid_view_name"), None)

    def test_list_data(self):
        # TODO: Implement this test
        pass

    def test_get_bq_table(self):
        # TODO: Implement this test
        pass

    def test_collectBQError(self):
        # TODO: Implement this test
        pass

    def test_insertDataFrameToBQ(self):
        # TODO: Implement this test
        pass

    def test_checkFirstLoad(self):
        # TODO: Implement this test
        pass

    def test_list_model_log(self):
        # TODO: Implement this test
        pass

    def test_select_actual_action(self):
        # TODO: Implement this test
        pass

    def test_retrive_next_data_from_view(self):
        # TODO: Implement this test
        pass

    def test_retrive_first_data_from_view(self):
        # TODO: Implement this test
        pass

    def test_retrive_one_row_from_view_to_gen_df_schema(self):
        # TODO: Implement this test
        pass

    def test_add_acutal_action_to_df_at_next(self):
        # TODO: Implement this test
        pass

    def test_check_duplicate_ID(self):
        # TODO: Implement this test
        pass

    def test_insert_data_to_BQ_data_frame(self):
        # TODO: Implement this test
        pass

    def test_run_StoreProcedure_To_Merge_Temp_Main_and_Truncate_Transaction(self):
        # TODO: Implement this test
        pass

if __name__ == '__main__':
    unittest.main()
