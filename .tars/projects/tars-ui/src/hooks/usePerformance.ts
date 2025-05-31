import { useQuery } from '@tanstack/react-query';
import { mockTarsApi } from '../services/tarsApi';

export const usePerformance = () => {
  return useQuery({
    queryKey: ['performance'],
    queryFn: mockTarsApi.getPerformanceMetrics,
    refetchInterval: 2000, // Refetch every 2 seconds for real-time metrics
  });
};
