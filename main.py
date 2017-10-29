from object_net import object_net_components
from object_net import object_net_writer
from object_net import padder
import configargparse
import math
import prime_factors
import random
import tensorflow as tf
import tf_utils


def main():
    # Handle program arguments
    parser = configargparse.ArgParser()
    parser.add_argument(
        "--config", is_config_file=True, default="./object_net.ini", help="Path of ini configuration file")
    parser.add_argument("--hidden_vector_length", type=int, default=64)
    parser.add_argument("--fully_connected_sizes", type=str, default="256,256")
    parser.add_argument("--log_normalize", type=bool, default=False)
    tf_utils.generic_runner.GenericRunner.add_arguments(parser)
    tf_utils.data_holder.add_arguments(parser)
    prime_factors.add_arguments(parser)
    args = parser.parse_args()

    # Generate data
    print("Generating data...")
    tree_type = prime_factors.get_prime_factor_tree_type()
    trees = prime_factors.get_trees(args)
    if args.log_normalize:
        [prime_factors.log_normalise_tree(tree) for tree in trees]
    arrays = [list(tree_type.get_state_output_pairs(tree)) for tree in trees]
    random.shuffle(arrays)
    padded_arrays = padder.PaddedData.from_unpadded(arrays)
    data_holder = tf_utils.data_holder.DataHolder(
        args,
        get_data_fn=lambda i: padded_arrays[i],
        data_length=len(padded_arrays))
    print("Done")

    # Define graph
    truth_padded_data = padder.PlaceholderPaddedData()

    with tf.variable_scope("truth_initial_hidden_vector_input"):
        truth_initial_hidden_vector_input = tf.reshape(
            tf.slice(truth_padded_data.outputs_padded, [0, 1, 0], [-1, 1, 1]),
            [-1, 1])

    def get_object_net_writer(training: bool) -> object_net_writer.ObjectNetWriter:
        return object_net_writer.ObjectNetWriter(
            truth_padded_data,
            truth_initial_hidden_vector_input,
            object_type=tree_type,
            training=training,
            hidden_vector_network=object_net_components.LstmHiddenVectorNetwork(
                args.hidden_vector_length,
                num_layers=4,
                hidden_vector_combiner=object_net_components.AdditionHiddenVectorCombiner()))

    object_net = get_object_net_writer(training=True)
    object_net_test = get_object_net_writer(training=False)

    tf.summary.scalar("object_net/cost", object_net.cost)
    optimizer = tf.train.AdamOptimizer().minimize(object_net.cost)

    # Run training
    def train_step(session, step, training_input, all_summaries, summary_writer):
        _, all_summaries = session.run(
            [optimizer, all_summaries],
            truth_padded_data.get_feed_dict(training_input))

        summary_writer.add_summary(all_summaries, step)

    def test_step(session, step, testing_input, all_summaries, summary_writer):
        cost_result, all_summaries = session.run(
            [object_net.cost, all_summaries],
            truth_padded_data.get_feed_dict(testing_input))

        summary_writer.add_summary(all_summaries, step)

        print("Test cost at step %d: %f" % (step, cost_result))

        show_examples(session, testing_input)

    def show_examples(session, model_input):
        # Limit to 10 inputs
        model_input = [x[:10] for x in model_input]

        generated_states_padded, \
            generated_outputs_padded, \
            generated_outputs_counts_padded, \
            generated_step_counts, \
            current_initial_hidden_vector_input = session.run(
                [
                    object_net_test.generated_states_padded,
                    object_net_test.generated_outputs_padded,
                    object_net_test.generated_outputs_counts_padded,
                    object_net_test.generated_step_counts,
                    truth_initial_hidden_vector_input],
                truth_padded_data.get_feed_dict(model_input))

        copied_testing_input = padder.PaddedData(
            generated_step_counts, generated_outputs_counts_padded, generated_states_padded, generated_outputs_padded)
        unpadded = padder.unpad(copied_testing_input)

        def try_array_to_tree(_array):
            try:
                return tree_type.get_value_from_state_output_pairs(_array)
            except StopIteration:
                return prime_factors.create_tree(-1, None, None)

        generated_trees = [try_array_to_tree(array) for array in unpadded]

        print("Trees:")
        for tree, number in list(zip(generated_trees, current_initial_hidden_vector_input)):
            if args.log_normalize:
                number = math.pow(math.e, number)
                prime_factors.log_unnormalise_tree(tree)

            print("%d -> %s" % (round(number), tree))

        print("Raw unpadded data:")
        [print(list(unpadded)) for unpadded in padder.unpad(copied_testing_input)]

        print("Lengths:")
        print([len(list(unpadded)) for unpadded in padder.unpad(copied_testing_input)])

    runner = tf_utils.generic_runner.GenericRunner.from_args(args, "prime_factor_object_net")
    runner.set_data_holder(data_holder)
    runner.set_test_step(test_step)
    runner.set_train_step(train_step)
    runner.run()


if __name__ == "__main__":
    main()
