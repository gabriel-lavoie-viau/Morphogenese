import sys, os
os.environ["CUDA_VISIBLE_DEVICES"]="1"


"""Generate melodies from a trained checkpoint of a melody RNN model."""
import ast
import os
import time
import math
from magenta.models.melody_rnn import melody_rnn_config_flags
from magenta.models.melody_rnn import melody_rnn_model
from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator
from magenta.models.shared import sequence_generator_bundle
import note_seq
from note_seq.protobuf import generator_pb2
from note_seq.protobuf import music_pb2
import tensorflow.compat.v1 as tf

# Prevent tf to grab all of the GPU memory (tf v.1)
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
sess = tf.Session(config=config)

# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.set_visible_devices(gpus[0], 'GPU')
#     # tf.config.experimental.set_memory_growth(gpus[0], True)
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)

class melody_generator:

    def __init__(self, bundle_file=None):
        tf.disable_v2_behavior()

        self.generator = None

        self.FLAGS = tf.app.flags.FLAGS
        tf.app.flags.DEFINE_string(
            'run_dir', None,
            'Path to the directory where the latest checkpoint will be loaded from.')
        tf.app.flags.DEFINE_string(
            'checkpoint_file', None,
            'Path to the checkpoint file. run_dir will take priority over this flag.')
        tf.app.flags.DEFINE_string(
            'bundle_file', './bundles/basic_rnn.mag',
            'Path to the bundle file. If specified, this will take priority over '
            'run_dir and checkpoint_file, unless save_generator_bundle is True, in '
            'which case both this flag and either run_dir or checkpoint_file are '
            'required')
        tf.app.flags.DEFINE_boolean(
            'save_generator_bundle', False,
            'If true, instead of generating a sequence, will save this generator as a '
            'bundle file in the location specified by the bundle_file flag')
        tf.app.flags.DEFINE_string(
            'bundle_description', None,
            'A short, human-readable text description of the bundle (e.g., training '
            'data, hyper parameters, etc.).')
        tf.app.flags.DEFINE_string(
            'output_dir', '/media/gabriel/Extra/melody_rnn/generated',
            'The directory where MIDI files will be saved to.')
        tf.app.flags.DEFINE_integer(
            'num_outputs', 1,
            'The number of melodies to generate. One MIDI file will be created for '
            'each.')
        tf.app.flags.DEFINE_integer(
            'num_steps', 128,
            'The total number of steps the generated melodies should be, priming '
            'melody length + generated steps. Each step is a 16th of a bar.')
        tf.app.flags.DEFINE_string(
            'primer_melody', '', 'A string representation of a Python list of '
            'note_seq.Melody event values. For example: '
            '"[60, -2, 60, -2, 67, -2, 67, -2]". If specified, this melody will be '
            'used as the priming melody. If a priming melody is not specified, '
            'melodies will be generated from scratch.')
        tf.app.flags.DEFINE_string(
            'primer_midi', '',
            'The path to a MIDI file containing a melody that will be used as a '
            'priming melody. If a primer melody is not specified, melodies will be '
            'generated from scratch.')
        tf.app.flags.DEFINE_float(
            'qpm', None,
            'The quarters per minute to play generated output at. If a primer MIDI is '
            'given, the qpm from that will override this flag. If qpm is None, qpm '
            'will default to 120.')
        tf.app.flags.DEFINE_float(
            'temperature', 1.0,
            'The randomness of the generated melodies. 1.0 uses the unaltered softmax '
            'probabilities, greater than 1.0 makes melodies more random, less than 1.0 '
            'makes melodies less random.')
        tf.app.flags.DEFINE_integer(
            'beam_size', 1,
            'The beam size to use for beam search when generating melodies.')
        tf.app.flags.DEFINE_integer(
            'branch_factor', 1,
            'The branch factor to use for beam search when generating melodies.')
        tf.app.flags.DEFINE_integer(
            'steps_per_iteration', 1,
            'The number of melody steps to take per beam search iteration.')
        tf.app.flags.DEFINE_string(
            'log', 'INFO',
            'The threshold for what messages will be logged DEBUG, INFO, WARN, ERROR, '
            'or FATAL.')

        if bundle_file == None:
            print("\nNo bundle file path provided. Using by default: " + self.FLAGS.bundle_file)
        else:
            print("\nLoading bundle file: " + bundle_file)
            self.FLAGS.bundle_file = bundle_file

        self.primer_steps_scaler    = 1.0
        self.num_steps_scaler       = 1.0 
        self.primer_toggle          = False
        self.voice_name             = 'voice0'


    def get_checkpoint(self):
        """Get the training dir or checkpoint path to be used by the model."""
        if ((self.FLAGS.run_dir or self.FLAGS.checkpoint_file) and
            self.FLAGS.bundle_file and not self.FLAGS.save_generator_bundle):
            raise sequence_generator.SequenceGeneratorError(
                'Cannot specify both bundle_file and run_dir or checkpoint_file')
        if self.FLAGS.run_dir:
            train_dir = os.path.join(os.path.expanduser(self.FLAGS.run_dir), 'train')
            return train_dir
        elif self.FLAGS.checkpoint_file:
            return os.path.expanduser(self.FLAGS.checkpoint_file)
        else:
            return None

    def get_bundle(self):
        """Returns a generator_pb2.GeneratorBundle object based read from bundle_file.

        Returns:
            Either a generator_pb2.GeneratorBundle or None if the bundle_file flag is
            not set or the save_generator_bundle flag is set.
        """
        if self.FLAGS.save_generator_bundle:
            return None
        if self.FLAGS.bundle_file is None:
            return None
        bundle_file = os.path.expanduser(self.FLAGS.bundle_file)
        return sequence_generator_bundle.read_bundle_file(bundle_file)


    def run_with_flags(self, generator):
        """Generates melodies and saves them as MIDI files.

        Uses the options specified by the flags defined in this module.

        Args:
            generator: The MelodyRnnSequenceGenerator to use for generation.
        """
        if not self.FLAGS.output_dir:
            tf.logging.fatal('--output_dir required')
            return
        self.FLAGS.output_dir = os.path.expanduser(self.FLAGS.output_dir)

        primer_midi = None
        if self.FLAGS.primer_midi:
            primer_midi = os.path.expanduser(self.FLAGS.primer_midi)
        if not tf.gfile.Exists(self.FLAGS.output_dir):
            tf.gfile.MakeDirs(self.FLAGS.output_dir)

        primer_sequence = None
        qpm = self.FLAGS.qpm if self.FLAGS.qpm else note_seq.DEFAULT_QUARTERS_PER_MINUTE
        if self.FLAGS.primer_melody:
            primer_melody = note_seq.Melody(ast.literal_eval(self.FLAGS.primer_melody))
            primer_sequence = primer_melody.to_sequence(qpm=qpm)
        elif primer_midi: # This the one
            if (self.primer_toggle == True): 
                # Using primer from user input
                primer_sequence = note_seq.midi_file_to_sequence_proto(primer_midi)
            else: 
                # Using primer from end of last sequence
                midi_filename = self.voice_name + '_midi_score.mid'
                midi_filepath = os.path.join('./generated', midi_filename)
                primer_sequence = note_seq.midi_file_to_sequence_proto(midi_filepath)
                seq_end = max(n.end_time for n in primer_sequence.notes)
                seq_start = seq_end - (seq_end * 0.25)
                primer_sequence = note_seq.extract_subsequence(primer_sequence, seq_start, seq_end)

            if primer_sequence.tempos and primer_sequence.tempos[0].qpm:
                qpm = primer_sequence.tempos[0].qpm
        else:
            tf.logging.warning(
                'No priming sequence specified. Defaulting to a single middle C.')
            primer_melody = note_seq.Melody([60])
            primer_sequence = primer_melody.to_sequence(qpm=qpm)

        # Derive the total number of seconds to generate based on the QPM of the
        # priming sequence and the num_steps flag.
        seconds_per_step = 60.0 / qpm / generator.steps_per_quarter
        total_seconds = self.FLAGS.num_steps * seconds_per_step

        # Specify start/stop time for generation based on starting generation at the
        # end of the priming sequence and continuing until the sequence is num_steps
        # long.
        generated_time = 0
        generator_options = generator_pb2.GeneratorOptions()
        if primer_sequence:

            input_sequence = primer_sequence
            # Set the start time to begin on the next step after the last note ends.
            if primer_sequence.notes:
                primer_end_time = max(n.end_time for n in primer_sequence.notes) # End time of last note of primer
                generated_time = total_seconds - primer_end_time
                ending_time = primer_end_time + generated_time
                generate_section = generator_options.generate_sections.add( start_time=primer_end_time + seconds_per_step,
                                                                            end_time=ending_time)


                # primer_end_time = max(n.end_time for n in primer_sequence.notes) # End time of last note of primer
                # generated_time = total_seconds - primer_end_time
                # ending_time = primer_end_time + (generated_time * self.num_steps_scaler)
                # generate_section = generator_options.generate_sections.add( start_time=primer_end_time + seconds_per_step,
                #                                                             end_time=ending_time)
            else:
                print('\nNo primer sequence received')
                primer_end_time = 0
                generate_section = generator_options.generate_sections.add( start_time=primer_end_time + seconds_per_step,
                                                                            end_time=total_seconds)

            if generate_section.start_time >= generate_section.end_time:
                tf.logging.fatal(
                    'Priming sequence is longer than the total number of steps '
                    'requested: Priming sequence length: %s, Generation length '
                    'requested: %s',
                    generate_section.start_time, total_seconds)
                return
        else:
            input_sequence = music_pb2.NoteSequence()
            input_sequence.tempos.add().qpm = qpm
            generate_section = generator_options.generate_sections.add(
                start_time=0,
                end_time=total_seconds)
        generator_options.args['temperature'].float_value = self.FLAGS.temperature
        generator_options.args['beam_size'].int_value = self.FLAGS.beam_size
        generator_options.args['branch_factor'].int_value = self.FLAGS.branch_factor
        generator_options.args['steps_per_iteration'].int_value = self.FLAGS.steps_per_iteration
        tf.logging.debug('input_sequence: %s', input_sequence)
        tf.logging.debug('generator_options: %s', generator_options)

        # Make the generate request num_outputs times and save the output as midi
        # files.
        date_and_time = time.strftime('%Y-%m-%d_%H%M%S')
        digits = len(str(self.FLAGS.num_outputs))
        for i in range(self.FLAGS.num_outputs):
            generated_sequence = generator.generate(input_sequence, generator_options)

            # Restrain melody between melody_min and melody_max
            melody_min = 40
            melody_max = 64
            for n in generated_sequence.notes:
                if n.pitch < melody_min:
                    n.pitch = n.pitch + math.ceil((melody_min - n.pitch) / 12) * 12
                if n.pitch > melody_max:
                    n.pitch = n.pitch - math.ceil((n.pitch - melody_max) / 12) * 12

            seq_start = primer_end_time - (primer_end_time * self.primer_steps_scaler)
            seq_end = primer_end_time + (generated_time * self.num_steps_scaler)
            generated_sequence = note_seq.extract_subsequence(generated_sequence, seq_start, seq_end)
            # seq_start = primer_end_time - (primer_end_time * self.primer_steps_scaler)
            # seq_end = max(n.end_time for n in generated_sequence.notes)
            # generated_sequence = note_seq.extract_subsequence(generated_sequence, seq_start, seq_end)

            # midi_filename = '%s_%s.mid' % (date_and_time, str(i + 1).zfill(digits))
            midi_filename = self.voice_name + '_midi_score.mid'
            midi_filepath = os.path.join('./generated', midi_filename)
            note_seq.sequence_proto_to_midi_file(generated_sequence, midi_filepath)
        
        return midi_filename


    def load(self):
        """Saves bundle or runs generator based on flags."""
        tf.logging.set_verbosity(self.FLAGS.log)
        
        bundle = self.get_bundle()

        if bundle:
            config_id = bundle.generator_details.id
            config = melody_rnn_model.default_configs[config_id]
            config.hparams.parse(self.FLAGS.hparams)
        else:
            config = melody_rnn_config_flags.config_from_flags()

        self.generator = melody_rnn_sequence_generator.MelodyRnnSequenceGenerator(
            model=melody_rnn_model.MelodyRnnModel(config),
            details=config.details,
            steps_per_quarter=config.steps_per_quarter,
            checkpoint=self.get_checkpoint(),
            bundle=bundle)


    def predict(self,   
                output_dir='../melody_generator/generated', 
                num_outputs=1, 
                num_steps=64, 
                temperature=1.0, 
                primer_midi='', 
                primer_steps_scaler=1.0, 
                num_steps_scaler=1.0,
                primer_toggle=False,
                voice_name='voice0'):

        self.FLAGS.output_dir   = output_dir
        self.FLAGS.num_outputs  = num_outputs
        self.FLAGS.num_steps    = num_steps
        self.FLAGS.temperature  = temperature
        self.FLAGS.primer_midi  = primer_midi
        self.primer_steps_scaler= primer_steps_scaler
        self.num_steps_scaler   = num_steps_scaler
        self.primer_toggle      = primer_toggle
        self.voice_name         = voice_name

        if self.FLAGS.save_generator_bundle:
            bundle_filename = os.path.expanduser(self.FLAGS.bundle_file)
            if self.FLAGS.bundle_description is None:
                tf.logging.warning('No bundle description provided.')
            tf.logging.info('Saving generator bundle to %s', bundle_filename)
            self.generator.create_bundle_file(bundle_filename, self.FLAGS.bundle_description)
        else:
            midi_filename = self.run_with_flags(self.generator)
            midi_filepath = os.path.join('../melody_generator/generated', midi_filename)
        # print('\nWrote', self.FLAGS.num_outputs, 'MIDI files to', self.FLAGS.output_dir, '\n')

        return midi_filepath