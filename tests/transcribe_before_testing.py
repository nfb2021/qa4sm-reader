from test_transcriber.transcribe import TestDataTranscriber

transcriber = TestDataTranscriber()

to_be_transcribed = transcriber.copy_nc_files()
unsuccesfull = []
for f in to_be_transcribed:
    transcriber.transcribe(f)
