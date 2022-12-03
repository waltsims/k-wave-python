classdef TestRecorder < handle
   properties
      record_filename = ""
      records = containers.Map
      step = 0
   end
   methods
       
       function obj = TestRecorder(record_filename)
           obj.record_filename = record_filename;
       end
       
       function obj = recordExpectedValue(obj, name, value)
           key = ['step_' num2str(obj.step) '___' name];
           obj.records(key) = value;
       end
       
       function obj = increment(obj)
           obj.step = obj.step + 1;
       end
       
      function obj = saveRecordsToDisk(obj)
          k = keys(obj.records) ;
          val = values(obj.records) ;
          for i = 1:length(obj.records)
              curr_key = k{i};
              curr_val = val{i};
              eval([curr_key, '=curr_val;']);
          end
          total_steps = obj.step;
          save(obj.record_filename, k{:}, 'total_steps');
      end
      
   end
end
